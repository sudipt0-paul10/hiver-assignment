#!/usr/bin/env python3
"""Sample the ~200-item hand-labelled GOLDEN evaluation set from the TEST window.

Three modes, no UI, no API, no model, no labels:

    python scripts/sample_golden.py --build     # write the sampling artefacts
    python scripts/sample_golden.py --verify    # 11 structural checks vs twcs.csv
    python scripts/sample_golden.py --validate data/golden/golden_items.csv

This is *sampling only*. Nothing here predicts an intent, an escalation, or a
reply; the strata below are retrieval devices built from observable text
properties, never labels and never expected answers (see docs/golden_set.md §4).

Relationship to the pilot
-------------------------
`scripts/annotate_pilot.py` built the locked 40-item codebook pilot from **dev**.
This script reuses that infrastructure directly - `profile.stream_structure`,
`profile.build_threads`, `profile.SPLITS`, `profile.strip_signature`,
`annotate_pilot._dedup`, `annotate_pilot.PILOT_SEEDS`, `annotate_pilot.COLUMNS`
and `annotate_pilot.validate` - rather than reimplementing any of it. Four
things differ, each deliberate and documented in docs/golden_set.md §7:

1. **Split is TEST**, not dev, and is not overridable. The golden set is the
   evaluation set (D18); the pilot avoided test precisely so that the codebook
   revision would not tune against it (D7).
2. **Core is drawn FIRST.** In the pilot the biased strata were drawn before
   `core`, which left `core` a random sample of "the pool minus 40
   non-randomly-removed items". Drawing core first makes it an honest simple
   random sample of the deduplicated test pool; the targeted strata, which are
   intentionally biased anyway, take what is left.
3. **The annotation CSV carries the exact original tweet text**, byte-for-byte
   from twcs.csv - not the D10-normalised text the pilot CSV showed. D10 fixes
   normalisation for *retrieval and measurement* and states the raw text is what
   the generator sees; the annotator should judge the same string the system
   will. It also matters for labelling: the `public_pii` escalation reason and
   the `existing_case_followup` state are partly carried by mentions and URLs
   that normalisation deletes.
4. **Near-duplicate removal is implemented** (word-3-gram Jaccard). D6 and D18
   require it; the pilot only removed exact normalised duplicates.

Nothing in outputs/pilot/ is read for anything except the exclusion list, and
nothing there is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate_pilot as AP  # noqa: E402
import profile as P  # noqa: E402

GOLDEN_SEED = 20171120          # the first day of the test window; arbitrary but recorded
SPLIT = "test"                  # not a flag: the golden set is the evaluation set (D18)
NEAR_DUP_JACCARD = 0.80         # word-3-gram Jaccard at or above this = near-duplicate
SHINGLE_N = 3

# ---------------------------------------------------------------------------
# Strata. Every mask below is computed from OBSERVABLE TEXT ONLY. A stratum is
# a retrieval device: an item drawn as `targeted:billing_subscription` is not
# expected to be labelled billing_subscription, and disagreement is a finding,
# not an error. Quotas are deliberately non-proportional - see docs/golden_set.md.
# ---------------------------------------------------------------------------
N_CORE = 160

# Boundary pairs, chosen from the confusions the 40-item pilot actually exposed
# (docs/taxonomy.md 9.2-9.4): BR-11 account/device, BR-3 availability/playback,
# BR-4 playback/device, BR-6 feedback/device, BR-2 billing/plans.
BOUNDARY_PAIRS = [
    ("account_access", "app_device_technical"),
    ("content_availability", "playback_playlist"),
    ("playback_playlist", "app_device_technical"),
    ("product_feature_feedback", "app_device_technical"),
    ("billing_subscription", "plans_eligibility"),
]
N_PER_BOUNDARY_PAIR = 2          # 5 pairs x 2 = 10

# Escalation enrichment. The proxies are profile.MUST_PROXY (already used to
# quote MUST prevalence throughout docs/) plus two text markers defined here.
# They enrich the *chance* of escalation-positive items; they do not assign
# escalation, which is a human judgement under the D9 policy.
N_ESCALATION = {
    "legal_rights": 1,
    "safety_abuse": 1,
    "security": 2,
    "existing_case": 2,
    "failed_self_service": 2,
    "account_access": 2,
    "payment_dispute": 2,
}                                # = 12

N_VAGUE = 4                      # very short / context-free -> BR-8, BR-10, other_unclear

TARGETED_INTENTS = [             # 7 seeded families x 2 = 14; other_unclear is
    "account_access",            # covered by the vague stratum, which is what
    "billing_subscription",      # BR-10 items actually look like in the data
    "plans_eligibility",
    "content_availability",
    "playback_playlist",
    "app_device_technical",
    "product_feature_feedback",
]
N_PER_TARGETED = 2
# 160 core + 10 boundary + 12 escalation + 4 vague + 14 targeted = 200

# Text markers, documented as proxies. EXISTING_CASE mirrors the pilot's
# follow-up regex exactly; FAILED_SELF_SERVICE is new and is the only proxy in
# this file that has no precedent elsewhere in the repo.
EXISTING_CASE = (
    r"\b(?:dm(?:'?d| sent| you)|sent (?:you )?a dm|check your dm|answer my dm)\b"
)
FAILED_SELF_SERVICE = (
    r"\b(?:tried everything|already tried|i.?ve tried|tried (?:that|this|it) (?:already|too)|"
    r"still (?:not|isn.?t|doesn.?t|won.?t|no)|reinstall|re-?installed|restarted|rebooted|"
    r"uninstall|no (?:one|body) (?:has )?(?:replied|responded|answered)|no response|"
    r"third time|3rd time|second time|2nd time|weeks? (?:now|and)|no luck|nothing works)\b"
)


# ---------------------------------------------------------------------------
# near-duplicate detection: word-3-gram Jaccard, greedy keep-first
# ---------------------------------------------------------------------------
def shingles(text: str, n: int = SHINGLE_N) -> frozenset[str]:
    toks = text.split()
    if len(toks) < n:
        return frozenset([" ".join(toks)]) if toks else frozenset()
    return frozenset(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def near_dup_matches(cands: list[frozenset[str]], refs: list[frozenset[str]],
                     thresh: float) -> list[int]:
    """Indices of `cands` that are near-duplicates of any shingle set in `refs`.

    Inverted index over shingles so only pairs sharing at least one 3-gram are
    compared - an exhaustive O(n*m) Jaccard over ~6k x ~15k texts would be
    pointlessly slow, and any pair below the threshold shares few shingles.
    """
    index: dict[str, list[int]] = defaultdict(list)
    for j, s in enumerate(refs):
        for sh in s:
            index[sh].append(j)
    hits = []
    for i, s in enumerate(cands):
        seen: set[int] = set()
        for sh in s:
            seen.update(index.get(sh, ()))
        if any(_jaccard(s, refs[j]) >= thresh for j in seen):
            hits.append(i)
    return hits


def near_dup_collapse(sets: list[frozenset[str]], thresh: float) -> np.ndarray:
    """Boolean keep-mask: greedy first-wins collapse of near-duplicates within a set."""
    index: dict[str, list[int]] = defaultdict(list)
    keep = np.ones(len(sets), dtype=bool)
    for i, s in enumerate(sets):
        seen: set[int] = set()
        for sh in s:
            seen.update(index.get(sh, ()))
        if any(_jaccard(s, sets[j]) >= thresh for j in seen):
            keep[i] = False
            continue
        for sh in s:
            index[sh].append(i)
    return keep


# ---------------------------------------------------------------------------
# raw text: read back with NA-coercion disabled so the CSV carries the exact
# source bytes. pandas' default na_values would silently turn a tweet reading
# "nan", "null" or "N/A" into a float NaN.
# ---------------------------------------------------------------------------
def raw_text_lookup(csv_path: str, ids: Iterable[int], chunksize: int) -> dict[int, str]:
    want = pd.Index(list(ids))
    out: dict[int, str] = {}
    for chunk in pd.read_csv(csv_path, usecols=["tweet_id", "text"], chunksize=chunksize,
                             dtype={"tweet_id": "int64", "text": "str"},
                             keep_default_na=False, na_filter=False):
        hit = chunk[chunk["tweet_id"].isin(want)]
        for tid, txt in zip(hit["tweet_id"], hit["text"]):
            out[int(tid)] = txt
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# pool
# ---------------------------------------------------------------------------
def build_pool(csv_path: str, brand: str, chunksize: int, excl: dict[str, Any],
               pilot_texts: list[str]) -> tuple[pd.DataFrame, dict[str, Any], set[str]]:
    struct = P.stream_structure(csv_path, [brand], chunksize)
    root, _ = P.build_threads(struct["tweet_id"], struct["parent"])
    is_root = root == np.arange(len(root))
    threads = np.unique(root[struct["masks"][brand]])
    keep = np.isin(root, threads)
    text = P.stream_text(csv_path, struct["tweet_id"][keep], chunksize)
    base = pd.DataFrame({
        "tweet_id": struct["tweet_id"][keep],
        "inbound": struct["inbound"][keep],
        "thread": root[keep],
        "is_root": is_root[keep],
    })
    frame = base.merge(text, on="tweet_id", how="inner")
    del struct, root, is_root, keep, base, text

    openers = frame[frame["is_root"] & frame["inbound"]].copy()
    tlo, thi = P.SPLITS["train"]
    train_customers = set(
        openers[(openers["created_at"] >= tlo) & (openers["created_at"] < thi)]["author_id"]
    )

    lo, hi = P.SPLITS[SPLIT]
    pool = openers[(openers["created_at"] >= lo) & (openers["created_at"] < hi)].copy()
    trace = {"test_openers_with_brand_reply": int(len(pool))}

    pool["clean"] = pool["text"].map(P.strip_signature)
    pool = pool[pool["clean"].str.len() > 0]
    trace["after_empty_normalised_text"] = int(len(pool))

    # --- pilot exclusion, before any dedup so a pilot customer can never win a slot
    n0 = len(pool)
    pool = pool[~pool["thread"].isin(set(excl["thread_ids"]))]
    trace["removed_pilot_thread"] = n0 - len(pool)
    n0 = len(pool)
    pool = pool[~pool["author_id"].isin(set(excl["customer_ids"]))]
    trace["removed_pilot_customer"] = n0 - len(pool)
    n0 = len(pool)
    pool = pool[~pool["tweet_id"].isin(set(excl["tweet_ids"]))]
    trace["removed_pilot_tweet"] = n0 - len(pool)

    # --- isolation + exact duplicates: the pilot's own rules.
    # Unrolled only so each step can be counted; asserted below to be identical
    # to annotate_pilot._dedup, so the two stages cannot silently diverge.
    pre_dedup = pool
    pool = pool.sort_values("tweet_id")
    n0 = len(pool)
    pool = pool.drop_duplicates(subset=["thread"], keep="first")
    trace["removed_multi_opener_thread"] = n0 - len(pool)
    n0 = len(pool)
    pool = pool.drop_duplicates(subset=["author_id"], keep="first")
    trace["removed_repeat_customer"] = n0 - len(pool)
    n0 = len(pool)
    pool = pool.drop_duplicates(subset=["clean"], keep="first")
    trace["removed_exact_normalised_duplicate"] = n0 - len(pool)

    assert set(pool["tweet_id"]) == set(AP._dedup(pre_dedup)["tweet_id"]), \
        "dedup diverged from annotate_pilot._dedup"
    del pre_dedup

    # --- near-duplicates (new; D6/D18 require them, the pilot never implemented them)
    sets = [shingles(t) for t in pool["clean"]]
    keep_mask = near_dup_collapse(sets, NEAR_DUP_JACCARD)
    trace["removed_near_duplicate_within_test"] = int((~keep_mask).sum())
    pool = pool[keep_mask]

    sets = [shingles(t) for t in pool["clean"]]
    pilot_sets = [shingles(t) for t in pilot_texts]
    hits = near_dup_matches(sets, pilot_sets, NEAR_DUP_JACCARD)
    trace["removed_near_duplicate_of_pilot_item"] = len(hits)
    if hits:
        pool = pool.drop(pool.index[hits])

    trace["pool_final"] = int(len(pool))
    return pool.reset_index(drop=True), trace, train_customers


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(csv_path: str, out_dir: str, brand: str, chunksize: int,
          pilot_dir: str) -> dict[str, Any]:
    excl_path = os.path.join(pilot_dir, "pilot_exclusions.json")
    with open(excl_path, encoding="utf-8") as fh:
        excl = json.load(fh)
    with open(os.path.join(pilot_dir, "pilot_items.csv"), encoding="utf-8-sig", newline="") as fh:
        pilot_texts = [r["text"] for r in csv.DictReader(fh)]

    pool, trace, train_customers = build_pool(csv_path, brand, chunksize, excl, pilot_texts)
    rng = np.random.default_rng(GOLDEN_SEED)

    seed_hits = {n: pool["text"].str.contains(p, case=False, regex=True)
                 for n, p in AP.PILOT_SEEDS.items()}
    n_families = sum(h.astype(int) for h in seed_hits.values())
    must_hits = {n: pool["text"].str.contains(p, case=False, regex=True)
                 for n, p in P.MUST_PROXY.items()}
    esc_masks = {
        "legal_rights": must_hits["legal_rights"],
        "safety_abuse": must_hits["safety"],
        "security": must_hits["security_privacy"],
        "existing_case": pool["text"].str.contains(EXISTING_CASE, case=False, regex=True),
        "failed_self_service": pool["text"].str.contains(FAILED_SELF_SERVICE, case=False, regex=True),
        "account_access": must_hits["account_access"],
        "payment_dispute": must_hits["payment_dispute"],
    }

    chosen: dict[int, str] = {}
    prob: dict[int, float] = {}
    draw_log: list[dict[str, Any]] = []

    def take(mask: pd.Series, k: int, label: str) -> None:
        cand = pool[mask & ~pool["tweet_id"].isin(list(chosen))]
        n = len(cand)
        got = min(k, n)
        draw_log.append({"stratum": label, "requested": k, "eligible_at_draw": int(n),
                         "drawn": int(got),
                         "inclusion_prob": round(got / n, 6) if n else None})
        if got <= 0:
            return
        idx = rng.choice(n, size=got, replace=False)
        for tid in cand.iloc[np.sort(idx)]["tweet_id"]:
            chosen[int(tid)] = label
            prob[int(tid)] = got / n

    # 1. CORE FIRST. This is the only unbiased stratum and the only one a
    #    population estimate may rest on; drawing it before the biased strata is
    #    what makes it a genuine simple random sample of the pool.
    take(pd.Series(True, index=pool.index), N_CORE, "core")
    # 2. scarcest enrichment next, so a rare stratum is not starved by a common one
    take(esc_masks["legal_rights"], N_ESCALATION["legal_rights"], "escalation:legal_rights")
    take(esc_masks["safety_abuse"], N_ESCALATION["safety_abuse"], "escalation:safety_abuse")
    for a, b in BOUNDARY_PAIRS:
        take(seed_hits[a] & seed_hits[b], N_PER_BOUNDARY_PAIR, f"boundary:{a}|{b}")
    for name in ("security", "existing_case", "failed_self_service",
                 "account_access", "payment_dispute"):
        take(esc_masks[name], N_ESCALATION[name], f"escalation:{name}")
    # 3. vague / context-free: what BR-8 and BR-10 items look like in the data
    take(pool["clean"].str.split().str.len() <= 6, N_VAGUE, "vague_unclear")
    # 4. clean single-family examples of each seeded intent
    for name in TARGETED_INTENTS:
        take(seed_hits[name] & (n_families == 1), N_PER_TARGETED, f"targeted:{name}")

    sel = pool[pool["tweet_id"].isin(list(chosen))].copy()
    sel["stratum"] = sel["tweet_id"].map(chosen)
    sel["inclusion_prob"] = sel["tweet_id"].map(prob)
    sel = sel.sample(frac=1.0, random_state=GOLDEN_SEED).reset_index(drop=True)
    sel["id"] = [f"G{i:03d}" for i in range(1, len(sel) + 1)]

    raw = raw_text_lookup(csv_path, sel["tweet_id"], chunksize)
    missing = [int(t) for t in sel["tweet_id"] if int(t) not in raw]
    if missing:
        raise SystemExit(f"raw text not found in {csv_path} for tweet_ids {missing}")

    os.makedirs(out_dir, exist_ok=True)
    items = os.path.join(out_dir, "golden_items.csv")
    with open(items, "w", newline="", encoding="utf-8") as fh:   # plain UTF-8, no BOM
        w = csv.DictWriter(fh, fieldnames=AP.COLUMNS)
        w.writeheader()
        for _, r in sel.iterrows():
            w.writerow({"id": r["id"], "text": raw[int(r["tweet_id"])],
                        **{c: "" for c in AP.COLUMNS[2:]}})

    xlsx = write_xlsx(items, os.path.join(out_dir, "golden_items.xlsx"))

    key_path = os.path.join(out_dir, "golden_key.json")
    key = {
        "_warning": "SAMPLING KEY - DO NOT OPEN BEFORE ANNOTATION IS COMPLETE. "
                    "Strata are retrieval devices, not labels, but seeing them "
                    "telegraphs an expected answer and corrupts the boundary "
                    "judgements this set exists to measure.",
        "seed": GOLDEN_SEED,
        "split": SPLIT,
        "items": {
            r["id"]: {
                "stratum": r["stratum"],
                "tweet_id": int(r["tweet_id"]),
                "thread_id": int(r["thread"]),
                "customer_id": str(r["author_id"]),
                "created_at": r["created_at"].isoformat(),
                "n_words_normalised": int(len(r["clean"].split())),
                "inclusion_prob": round(float(r["inclusion_prob"]), 6),
                "design_weight": round(1.0 / float(r["inclusion_prob"]), 4),
                "customer_also_in_train": bool(r["author_id"] in train_customers),
            }
            for _, r in sel.iterrows()
        },
    }
    with open(key_path, "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=2)

    manifest = {
        "codebook_version": json.load(open("docs/codebook.json", encoding="utf-8"))["version"],
        "brand": brand,
        "split_used": SPLIT,
        "split_window": list(P.SPLITS[SPLIT]),
        "seed": GOLDEN_SEED,
        "n_items": int(len(sel)),
        "n_core": int(sum(1 for v in chosen.values() if v == "core")),
        "n_targeted": int(sum(1 for v in chosen.values() if v != "core")),
        "pool_trace": trace,
        "dedup_rules": [
            "one opener per thread",
            "customer-disjoint within the golden set",
            "exact normalised-duplicate removal (D10 normalisation)",
            f"near-duplicate removal, word-{SHINGLE_N}-gram Jaccard >= {NEAR_DUP_JACCARD}",
            "near-duplicates of any pilot item removed",
        ],
        "pilot_exclusion": {
            "source": excl_path,
            "sha256": sha256_file(excl_path),
            "split": excl["split"],
            "n_threads": len(excl["thread_ids"]),
            "n_customers": len(excl["customer_ids"]),
            "n_tweets": len(excl["tweet_ids"]),
            "thread_ids": excl["thread_ids"],
            "customer_ids": excl["customer_ids"],
            "tweet_ids": excl["tweet_ids"],
        },
        "strata": dict(Counter(sel["stratum"])),
        "draw_log": draw_log,
        "customers_also_seen_in_train": int(sum(
            1 for _, r in sel.iterrows() if r["author_id"] in train_customers)),
        "text_form": "exact original twcs.csv text, unmodified",
        "brand_reply_included": False,
        "note": "Only the 'core' stratum is unbiased. Every other stratum is a "
                "deliberately enriched retrieval device built from observable text "
                "properties; strata are NOT labels and NOT expected answers. This "
                "set is not statistically representative of Spotify support traffic.",
        "files": {"items": items, "items_xlsx": xlsx, "key": key_path},
    }
    with open(os.path.join(out_dir, "golden_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# xlsx companion. Not decoration: the pilot's annotation round-trip through a
# spreadsheet corrupted non-ASCII text in 10 of 40 rows (docs/taxonomy.md 9.4),
# because a plain UTF-8 CSV without a BOM is read as ANSI by Excel and WPS. The
# CSV here stays the canonical, programmatically generated artefact; the xlsx is
# the surface to annotate on, and it cannot mojibake - xlsx is UTF-8 by spec.
# ---------------------------------------------------------------------------
def write_xlsx(csv_path: str, xlsx_path: str) -> str:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    wb = Workbook()
    ws = wb.active
    ws.title = "golden"
    ws.append(AP.COLUMNS)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append([r[c] for c in AP.COLUMNS])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.number_format = "@"
            # a tweet starting with "=", "+", "-" or "@" would otherwise be
            # stored as a formula and shown as an error by Excel
            if isinstance(c.value, str) and c.value[:1] in "=+-@":
                c.data_type = "s"
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    widths = {"id": 8, "text": 90, "notes": 40}
    for i, col in enumerate(AP.COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(col, 20)
    ws.freeze_panes = "C2"

    enums = {"intent": AP.INTENTS, "conversation_state": AP.CONVERSATION_STATE,
             "urgency": AP.LEVELS, "frustration": AP.LEVELS, "language": AP.LANGUAGE,
             "escalation": AP.ESCALATION, "escalation_reason": AP.REASONS,
             "confidence": AP.CONFIDENCE, "cannot_represent": ["y"]}
    last = len(rows) + 1
    for col, vals in enums.items():
        letter = get_column_letter(AP.COLUMNS.index(col) + 1)
        dv = DataValidation(type="list", formula1='"' + ",".join(vals) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{letter}2:{letter}{last}")
    wb.save(xlsx_path)
    return xlsx_path


def read_xlsx_rows(xlsx_path: str) -> list[dict[str, str]]:
    """Read the annotation workbook back as rows keyed by the locked schema.

    Blank cells arrive from openpyxl as None; they become "" so that every
    downstream consumer sees the same shape a CSV read would give.
    """
    from openpyxl import load_workbook
    ws = load_workbook(xlsx_path, read_only=True, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    header = [("" if c is None else str(c)).strip() for c in rows[0]][:len(AP.COLUMNS)]
    if header != AP.COLUMNS:
        raise SystemExit(f"{xlsx_path}: header is not the locked annotation schema\n"
                         f"  found:    {header}\n  expected: {AP.COLUMNS}")
    out = []
    for r in rows[1:]:
        if all(c is None or str(c).strip() == "" for c in r):
            continue
        out.append({c: ("" if v is None else str(v)) for c, v in zip(AP.COLUMNS, r)})
    return out


def export_annotations(out_dir: str, csv_path: str, chunksize: int) -> int:
    """xlsx -> data/golden/golden_annotated.csv, refusing to export corrupted text.

    The workbook is the human's working surface; every machine step downstream
    reads CSV, and the existing validator reads CSV. This is the one bridge
    between them, and it will not write a file whose text has drifted from the
    source dataset - which is exactly the failure the pilot hit.
    """
    xlsx = os.path.join(out_dir, "golden_items.xlsx")
    blank = os.path.join(out_dir, "golden_items.csv")
    rows = read_xlsx_rows(xlsx)
    with open(blank, encoding="utf-8", newline="") as fh:
        pristine = {r["id"]: r["text"] for r in csv.DictReader(fh)}

    problems = []
    if len(rows) != len(pristine):
        problems.append(f"row count {len(rows)} != {len(pristine)} in the pristine sample")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids in the workbook")
    if set(ids) != set(pristine):
        problems.append(f"id set changed: +{sorted(set(ids) - set(pristine))[:5]} "
                        f"-{sorted(set(pristine) - set(ids))[:5]}")
    drift = [r["id"] for r in rows if pristine.get(r["id"]) != r["text"]]
    if drift:
        problems.append(f"{len(drift)} rows whose text no longer matches the sample: {drift[:10]}")
    if problems:
        for p_ in problems:
            print(f"  ! {p_}", file=sys.stderr)
        raise SystemExit("export refused: the annotated workbook is not the sampled set")

    # second, independent integrity anchor: the source dataset itself
    key = json.load(open(os.path.join(out_dir, "golden_key.json"), encoding="utf-8"))["items"]
    raw = raw_text_lookup(csv_path, [v["tweet_id"] for v in key.values()], chunksize)
    src = [r["id"] for r in rows if r["text"] != raw.get(key[r["id"]]["tweet_id"])]
    if src:
        raise SystemExit(f"export refused: {len(src)} rows differ from twcs.csv: {src[:10]}")

    dest = os.path.join(out_dir, "golden_annotated.csv")
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=AP.COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in AP.COLUMNS})
    print(f"[export] {len(rows)} rows -> {dest}")
    print(f"[export] text verified against both {blank} and {csv_path}: "
          f"{len(rows)}/{len(rows)} byte-identical")
    return 0


def xlsx_text_mismatches(xlsx_path: str, expected: dict[str, str]) -> list[str]:
    """Ids whose xlsx text differs from the canonical CSV text.

    Guards the failure the pilot actually hit: 10 of 40 rows corrupted in a
    spreadsheet round-trip. If this ever fails, annotate the CSV directly.
    """
    from openpyxl import load_workbook
    ws = load_workbook(xlsx_path, read_only=True).active
    rows = ws.iter_rows(min_row=2, max_col=2, values_only=True)
    return [str(i) for i, t in rows if expected.get(str(i)) != (t or "")]


# ---------------------------------------------------------------------------
# verify: the 11 structural checks, run against twcs.csv and the pilot artefacts
# ---------------------------------------------------------------------------
def verify(csv_path: str, out_dir: str, brand: str, chunksize: int, pilot_dir: str) -> int:
    items = os.path.join(out_dir, "golden_items.csv")
    with open(items, "rb") as fh:
        head = fh.read(3)
    with open(items, encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        fieldnames = list(rdr.fieldnames or [])
        rows = list(rdr)
    key = json.load(open(os.path.join(out_dir, "golden_key.json"), encoding="utf-8"))["items"]
    man = json.load(open(os.path.join(out_dir, "golden_manifest.json"), encoding="utf-8"))
    excl = json.load(open(os.path.join(pilot_dir, "pilot_exclusions.json"), encoding="utf-8"))

    fails: list[str] = []

    def check(ok: bool, name: str, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  - ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    print("=== golden-set verification ===")
    check(len(rows) == man["n_items"] == 200,
          "1. row count", f"{len(rows)} rows, manifest says {man['n_items']}")
    ids = [r["id"] for r in rows]
    check(len(set(ids)) == len(ids), "2. ids unique", f"{len(set(ids))} distinct")
    check(set(ids) == set(key), "2b. ids match the key file")

    # rebuild the test-window opener universe independently of the sampler
    struct = P.stream_structure(csv_path, [brand], chunksize)
    root, _ = P.build_threads(struct["tweet_id"], struct["parent"])
    is_root = root == np.arange(len(root))
    threads = np.unique(root[struct["masks"][brand]])
    keep = np.isin(root, threads)
    text = P.stream_text(csv_path, struct["tweet_id"][keep], chunksize)
    frame = pd.DataFrame({"tweet_id": struct["tweet_id"][keep], "inbound": struct["inbound"][keep],
                          "thread": root[keep], "is_root": is_root[keep]}).merge(
        text, on="tweet_id", how="inner")
    lo, hi = P.SPLITS[SPLIT]
    op = frame[frame["is_root"] & frame["inbound"]]
    test_ids = set(op[(op["created_at"] >= lo) & (op["created_at"] < hi)]["tweet_id"].astype(int))
    sel_tids = {v["tweet_id"] for v in key.values()}
    outside = sorted(sel_tids - test_ids)
    check(not outside, "3. every item is a TEST-window opener with a brand reply",
          f"{len(outside)} outside" if outside else f"window {lo} -> {hi}")

    hit_t = sel_tids & set(excl["tweet_ids"])
    hit_th = {v["thread_id"] for v in key.values()} & set(excl["thread_ids"])
    hit_c = {v["customer_id"] for v in key.values()} & set(excl["customer_ids"])
    check(not (hit_t or hit_th or hit_c), "4. no pilot tweet / thread / customer",
          f"tweet={len(hit_t)} thread={len(hit_th)} customer={len(hit_c)}")

    th = [v["thread_id"] for v in key.values()]
    check(len(set(th)) == len(th), "5. one item per thread", f"{len(set(th))} threads")
    cu = [v["customer_id"] for v in key.values()]
    check(len(set(cu)) == len(cu), "6. customer-disjoint", f"{len(set(cu))} customers")

    raw = raw_text_lookup(csv_path, sorted(sel_tids), chunksize)
    bad = [r["id"] for r in rows if r["text"] != raw.get(key[r["id"]]["tweet_id"])]
    check(not bad, "7. exact text match against twcs.csv",
          f"{len(bad)} mismatched: {bad[:5]}" if bad else "200/200 byte-identical")

    outbound = sorted(sel_tids - set(frame.loc[frame["inbound"], "tweet_id"].astype(int)))
    check(not outbound, "8. no brand reply in the annotation artefact",
          "all selected tweets are inbound customer openers; no reply column exists"
          if not outbound else f"{len(outbound)} outbound rows")

    strat_vals = {v["stratum"] for v in key.values()} | {"stratum", "core", "boundary"}
    leaked = [r["id"] for r in rows
              if any((r.get(c) or "").strip() in strat_vals for c in AP.COLUMNS[2:])]
    check(fieldnames == AP.COLUMNS and "stratum" not in fieldnames and not leaked,
          "9. hidden strata absent from the annotation CSV",
          f"columns = {len(fieldnames)}, schema matches the locked annotation schema")

    blank = all(not (r.get(c) or "").strip() for r in rows for c in AP.COLUMNS[2:])
    check(blank, "10a. every annotation field is blank")
    check(head != b"\xef\xbb\xbf", "10b. CSV is plain UTF-8 with no BOM")
    xl = os.path.join(out_dir, "golden_items.xlsx")
    xl_bad = xlsx_text_mismatches(xl, {r["id"]: r["text"] for r in rows}) if os.path.exists(xl) else ["missing"]
    check(not xl_bad, "10c. xlsx companion round-trips the text exactly",
          f"{len(xl_bad)} mismatched: {xl_bad[:5]}" if xl_bad else f"{len(rows)}/{len(rows)} identical")

    print("\n=== stratum counts ===")
    for s, n in sorted(Counter(v["stratum"] for v in key.values()).items(),
                       key=lambda kv: (-kv[1], kv[0])):
        print(f"  {s:<48} {n:>3}")
    print(f"\n  core {man['n_core']}  |  targeted {man['n_targeted']}  |  total {man['n_items']}")
    print(f"  customers also seen in train (retrieval must exclude): "
          f"{man['customers_also_seen_in_train']}")
    short = [d for d in man["draw_log"] if d["drawn"] < d["requested"]]
    if short:
        print("\n  SHORTFALLS (stratum could not be filled from the pool):")
        for d in short:
            print(f"    {d['stratum']}: wanted {d['requested']}, eligible {d['eligible_at_draw']}, "
                  f"drawn {d['drawn']}")
    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--export", action="store_true",
                    help="golden_items.xlsx -> golden_annotated.csv, with text-integrity gates")
    ap.add_argument("--validate", metavar="CSV", help="post-annotation validation")
    ap.add_argument("--csv", default="twcs/twcs.csv")
    ap.add_argument("--brand", default="SpotifyCares")
    ap.add_argument("--out", default="data/golden")
    ap.add_argument("--pilot-dir", default="outputs/pilot")
    ap.add_argument("--chunksize", type=int, default=400_000)
    args = ap.parse_args()

    if args.validate:
        return AP.validate(args.validate,
                           keyfile=os.path.join(os.path.dirname(args.validate) or ".",
                                                "golden_key.json"))
    if args.export:
        return export_annotations(args.out, args.csv, args.chunksize)
    if args.verify:
        return verify(args.csv, args.out, args.brand, args.chunksize, args.pilot_dir)
    if not args.build:
        ap.error("pass --build, --export, --verify or --validate CSV")

    m = build(args.csv, args.out, args.brand, args.chunksize, args.pilot_dir)
    print(json.dumps({k: v for k, v in m.items() if k != "pilot_exclusion"}, indent=2))
    print(f"\n[golden] {m['n_items']} items from {m['split_used']} -> {m['files']['items']}",
          file=sys.stderr)
    print(f"[peak RSS] {P.peak_mb():.1f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
