#!/usr/bin/env python3
"""Build and validate the ~40-item taxonomy annotation pilot.

Two modes, no UI, no API, no model:

    python scripts/annotate_pilot.py --build
    python scripts/annotate_pilot.py --validate outputs/pilot/pilot_items.csv

``--build`` writes a CSV the annotator fills in by hand (Excel/LibreOffice or any
text editor) plus a JSON manifest recording exactly how every row was chosen.
``--validate`` reads the filled CSV back, rejects illegal values, and prints the
diagnostics the pilot exists to produce: boundary failures, thin classes, the
unclear rate, attribute ambiguity and escalation-policy ambiguity.

SPLIT CHOICE — read before changing
-----------------------------------
Default is **dev**, not test. The pilot's output is a *codebook revision*; if the
codebook is revised against test-window items, the evaluation set has been tuned
against, which is the contamination D7 forbids ("thresholds and prompts are tuned
on dev only; the test window is touched once"). The golden set still comes from
test — that is unchanged and is what keeps evaluation out-of-training. Dev is the
same brand, same reconstruction, and adjacent in time, so it is representative
for codebook purposes.

``--split test`` is available if the owner overrules this. In that case the
manifest still records every sampled thread and customer id, and those ids MUST
be excluded from the golden set — the exclusion file is written for that purpose.

WHAT THE ANNOTATOR MUST NOT SEE
-------------------------------
The historical brand reply is deliberately absent from the CSV (D21). Seeing it
would anchor the policy label onto observed behaviour and collapse D9's
separation of policy from behaviour. It is not written to any pilot artefact.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile as P  # noqa: E402
import taxonomy as T  # noqa: E402

PILOT_SEED = 913

# taxonomy.CANDIDATE_SEEDS predates the 9th class, and is deliberately NOT edited:
# the seed-based prevalences quoted in docs/taxonomy.md were produced with those
# exact seven patterns, and changing them would silently invalidate that document.
# The feature-feedback pattern below is the same one used to measure its 3.63%
# prevalence. These remain retrieval devices, never labels.
FEATURE_FEEDBACK_SEED = (
    r"\b(?:please add|add (?:an? )?(?:option|feature|button)|would (?:love|be nice|be great)|"
    r"wish (?:you|there|spotify)|suggestion|feature request|bring back|why can.?t (?:i|we|you)|"
    r"needs? an? (?:option|feature|way)|option to|there (?:should|needs to) be|"
    r"make it (?:possible|so)|hope you (?:add|fix))\b"
)
PILOT_SEEDS = {**T.CANDIDATE_SEEDS, "product_feature_feedback": FEATURE_FEEDBACK_SEED}

INTENTS = [
    "account_access", "billing_subscription", "plans_eligibility",
    "content_availability", "playback_playlist", "app_device_technical",
    "product_feature_feedback", "other_unclear",
]  # social_praise removed in codebook 0.2.0; social-only praise -> other_unclear
# Three values, per docs/codebook.json and docs/taxonomy.md. "unclear" covers
# messages giving no evidence either way about whether the issue is new.
CONVERSATION_STATE = ["opener", "existing_case_followup", "unclear"]
LEVELS = ["low", "normal", "high"]
ESCALATION = ["ESCALATE", "AUTO_OK"]
REASONS = [
    "none", "account_access", "payment_dispute", "security", "legal_rights",
    "safety_abuse", "public_pii", "existing_case", "failed_self_service",
    "high_frustration", "out_of_scope", "ambiguous",
]
LANGUAGE = ["english", "non_english", "unclear"]
CONFIDENCE = ["low", "medium", "high"]

# NOTE: `stratum` is deliberately NOT a column in the annotation CSV. Showing the
# annotator that an item was drawn as "targeted:account_access" telegraphs the
# expected answer and would corrupt exactly the boundary judgements the pilot is
# meant to measure. It is written to pilot_key.json and joined back at validation.
COLUMNS = [
    "id", "text",
    "intent", "conversation_state", "urgency", "frustration",
    "language",
    "escalation", "escalation_reason", "confidence", "notes", "cannot_represent",
]

# Targeted quotas. Deliberately NOT proportional — the pilot exists to stress the
# codebook, not to estimate prevalence. The core stratum is the only unbiased part.
QUOTAS = {
    "account_access": 2,
    "billing_subscription": 2,
    "plans_eligibility": 2,
    "content_availability": 2,
    "playback_playlist": 2,
    "app_device_technical": 2,
    "product_feature_feedback": 3,   # cleared the floor in the pilot (7/40)
}
N_BOUNDARY = 5      # items matching two or more seed families
N_AMBIGUOUS = 3     # very short / context-free / DM-followup markers
N_CORE = 17         # random, unbiased
# 15 targeted + 5 boundary + 3 ambiguous + 17 core = 40
# (social_praise's 3 targeted slots moved to core when the intent was removed;
#  its seed pattern is retained in PILOT_SEEDS for boundary detection and BR-5.)


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """One row per thread, customer-disjoint, exact normalised duplicates removed."""
    df = df.sort_values("tweet_id")
    df = df.drop_duplicates(subset=["thread"], keep="first")
    df = df.drop_duplicates(subset=["author_id"], keep="first")
    df = df.drop_duplicates(subset=["clean"], keep="first")
    return df


def build(csv_path: str, split: str, out_dir: str, brand: str, chunksize: int) -> dict[str, Any]:
    rng = np.random.default_rng(PILOT_SEED)
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
    lo, hi = P.SPLITS[split]
    pool = openers[(openers["created_at"] >= lo) & (openers["created_at"] < hi)].copy()
    pool["clean"] = pool["text"].map(P.strip_signature)
    pool = pool[pool["clean"].str.len() > 0]
    pool = _dedup(pool)

    seed_hits = {n: pool["text"].str.contains(p, case=False, regex=True)
                 for n, p in PILOT_SEEDS.items()}
    n_families = sum(h.astype(int) for h in seed_hits.values())

    chosen: dict[int, str] = {}

    def take(mask: pd.Series, k: int, label: str) -> None:
        cand = pool[mask & ~pool["tweet_id"].isin(list(chosen))]
        if len(cand) == 0 or k <= 0:
            return
        idx = rng.choice(len(cand), size=min(k, len(cand)), replace=False)
        for tid in cand.iloc[np.sort(idx)]["tweet_id"]:
            chosen[int(tid)] = label

    # 1. boundary first - scarcest and the pilot's main purpose
    take(n_families >= 2, N_BOUNDARY, "boundary")
    # 2. ambiguous / unclear candidates
    short = pool["clean"].str.split().str.len() <= 6
    followup = pool["text"].str.contains(
        r"\b(?:dm(?:'?d| sent| you)|sent (?:you )?a dm|check your dm|answer my dm)\b", case=False, regex=True)
    take(short | followup, N_AMBIGUOUS, "ambiguous")
    # 3. per-intent targeted quotas
    for name, k in QUOTAS.items():
        take(seed_hits[name] & (n_families == 1), k, f"targeted:{name}")
    # 4. unbiased core
    take(pd.Series(True, index=pool.index), N_CORE, "core")

    sel = pool[pool["tweet_id"].isin(list(chosen))].copy()
    sel["stratum"] = sel["tweet_id"].map(chosen)
    sel = sel.sample(frac=1.0, random_state=PILOT_SEED).reset_index(drop=True)  # shuffle so strata aren't guessable
    sel["id"] = [f"P{i:03d}" for i in range(1, len(sel) + 1)]

    os.makedirs(out_dir, exist_ok=True)
    items = os.path.join(out_dir, "pilot_items.csv")
    with open(items, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for _, r in sel.iterrows():
            w.writerow({"id": r["id"],
                        "text": re.sub(r"\s+", " ", r["clean"]).strip(),
                        **{c: "" for c in COLUMNS[2:]}})

    key = os.path.join(out_dir, "pilot_key.json")
    with open(key, "w", encoding="utf-8") as fh:
        json.dump({r["id"]: r["stratum"] for _, r in sel.iterrows()}, fh, indent=2)

    # Exclusion list: these threads/customers must never enter the golden set.
    excl = os.path.join(out_dir, "pilot_exclusions.json")
    with open(excl, "w", encoding="utf-8") as fh:
        json.dump({"split": split,
                   "thread_ids": sorted(int(x) for x in sel["thread"]),
                   "customer_ids": sorted(str(x) for x in sel["author_id"]),
                   "tweet_ids": sorted(int(x) for x in sel["tweet_id"])}, fh, indent=2)

    manifest = {
        "codebook_version": json.load(open("docs/codebook.json", encoding="utf-8"))["version"],
        "brand": brand,
        "split_used": split,
        "split_window": list(P.SPLITS[split]),
        "seed": PILOT_SEED,
        "pool_after_dedup": int(len(pool)),
        "n_items": int(len(sel)),
        "dedup_rules": ["one per thread", "customer-disjoint", "exact normalised duplicates removed"],
        "strata": dict(Counter(sel["stratum"])),
        "quotas_requested": {**QUOTAS, "boundary": N_BOUNDARY, "ambiguous": N_AMBIGUOUS, "core": N_CORE},
        "brand_reply_included": False,
        "note": "Targeted strata are deliberately non-proportional; only 'core' is unbiased. "
                "Seed families are retrieval devices, NOT labels or expected answers.",
        "files": {"items": items, "exclusions": excl, "key": key},
    }
    with open(os.path.join(out_dir, "pilot_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def validate(path: str, keyfile: str | None = None) -> int:
    """Validate an annotated CSV against the locked schema and print diagnostics.

    ``keyfile`` defaults to ``pilot_key.json`` beside the CSV. The golden-set
    sampler passes its own key here rather than duplicating this validator; that
    key nests its strata under an ``items`` object, so both shapes are accepted.
    """
    # utf-8-sig: Excel writes a UTF-8 BOM, which otherwise turns the first
    # column name into "\ufeffid". newline="": let csv handle CRLF itself.
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if keyfile is None:
        keyfile = os.path.join(os.path.dirname(path) or ".", "pilot_key.json")
    key = json.load(open(keyfile, encoding="utf-8")) if os.path.exists(keyfile) else {}
    if "items" in key:  # golden_key.json shape: {"_warning":..., "items": {id: {...}}}
        key = {k: v["stratum"] for k, v in key["items"].items()}
    allowed = {"intent": INTENTS, "conversation_state": CONVERSATION_STATE,
               "urgency": LEVELS, "frustration": LEVELS, "language": LANGUAGE,
               "escalation": ESCALATION, "escalation_reason": REASONS,
               "confidence": CONFIDENCE}
    errors, filled = [], 0
    for r in rows:
        if not any(r.get(c, "").strip() for c in allowed):
            continue
        filled += 1
        for col, vals in allowed.items():
            v = (r.get(col) or "").strip()
            # A blank escalation_reason on an AUTO_OK row means "none". Requiring the
            # literal word is a spreadsheet formality, not a labelling judgement, and
            # the AUTO_OK rule below already treats "" and "none" as equivalent.
            if not v and col == "escalation_reason" and (r.get("escalation") or "").strip() == "AUTO_OK":
                continue
            if not v:
                errors.append(f"{r['id']}: {col} empty")
            elif v not in vals:
                errors.append(f"{r['id']}: {col}='{v}' not in {vals}")
        if (r.get("escalation") or "").strip() == "AUTO_OK" and (r.get("escalation_reason") or "").strip() not in ("none", ""):
            errors.append(f"{r['id']}: AUTO_OK should carry reason 'none'")
        if (r.get("escalation") or "").strip() == "ESCALATE" and (r.get("escalation_reason") or "").strip() == "none":
            errors.append(f"{r['id']}: ESCALATE requires a reason other than 'none'")

    print(f"rows={len(rows)}  annotated={filled}  errors={len(errors)}")
    for e in errors[:40]:
        print("  !", e)
    if filled == 0:
        print("\nNothing annotated yet - fill the CSV, then re-run --validate.")
        return 1 if errors else 0

    done = [r for r in rows if (r.get("intent") or "").strip()]
    ic = Counter(r["intent"] for r in done)
    print("\n=== intent distribution ===")
    for i in INTENTS:
        n = ic.get(i, 0)
        flag = "  <-- THIN" if n <= 1 else ""
        print(f"  {i:<26} {n:>3}{flag}")

    unclear = ic.get("other_unclear", 0) / len(done)
    print(f"\nunclear rate {unclear:.1%}  (>15% => taxonomy inadequate, revise before golden set)")

    lowconf = [r for r in done if r.get("confidence") == "low"]
    print(f"low-confidence items: {len(lowconf)} ({len(lowconf)/len(done):.1%})")
    for r in lowconf:
        print(f"   {r['id']} [{r['intent']}] {r['text'][:72]}")
        if r.get("notes"):
            print(f"      note: {r['notes']}")

    cr = [r for r in done if (r.get("cannot_represent") or "").strip().lower() in ("y", "yes", "true", "1")]
    print(f"\nitems the codebook cannot represent: {len(cr)}")
    for r in cr:
        print(f"   {r['id']} {r['text'][:72]}  notes={r.get('notes','')}")

    print("\n=== targeted-stratum vs assigned intent (disagreement = boundary signal) ===")
    dis = 0
    for r in done:
        st = key.get(r["id"], "")
        if st.startswith("targeted:"):
            exp = st.split(":", 1)[1]
            if exp != r["intent"]:
                dis += 1
                print(f"   {r['id']} seeded as {exp} -> labelled {r['intent']}  conf={r.get('confidence')}")
    print(f"   targeted-stratum disagreements: {dis}"
          "   (seeds are crude; disagreement is informative, not an error)")

    print("\n=== boundary stratum outcomes (where confusion was expected) ===")
    for r in done:
        st = key.get(r["id"], "?")
        if st in ("boundary", "ambiguous"):
            print(f"   {r['id']} [{st}] -> {r['intent']} / conf={r.get('confidence')}")

    print("\n=== escalation ===")
    ec = Counter(r["escalation"] for r in done)
    print(f"   {dict(ec)}   ESCALATE rate = {ec.get('ESCALATE',0)/len(done):.1%}")
    print("   reasons:", dict(Counter(r["escalation_reason"] for r in done if r["escalation"] == "ESCALATE")))
    print("\n=== escalation by intent (does intent alone determine policy?) ===")
    by = {}
    for r in done:
        by.setdefault(r["intent"], Counter())[r["escalation"]] += 1
    for k, v in sorted(by.items()):
        mixed = " <-- MIXED: intent alone does not determine escalation" if len(v) > 1 else ""
        print(f"   {k:<26} {dict(v)}{mixed}")

    print("\n=== coherence: account-family reason on a non-account intent ===")
    print("   Advisory only. Escalation and intent are separate dimensions, so this is")
    print("   a review flag, not a prohibition. In the pilot, 3 of 4 flags were intent")
    print("   errors and 1 (P029) was a reason-code mismatch with a correct intent.")
    ACCOUNT_REASONS = {"account_access", "security"}
    flagged = [r for r in done
               if (r.get("escalation_reason") or "").strip() in ACCOUNT_REASONS
               and r["intent"] != "account_access"]
    if flagged:
        for r in flagged:
            print(f"   REVIEW {r['id']}  intent={r['intent']:<24} reason={r['escalation_reason']:<16} "
                  f"{r['text'][:52]}")
    print(f"   flagged for review: {len(flagged)}")

    print("\n=== language (attribute; never an intent, never an escalation reason) ===")
    lang = Counter(r["language"] for r in done)
    print(f"   {dict(lang)}")
    ne = [r for r in done if r["language"] != "english"]
    if ne:
        unc = sum(1 for r in ne if r["intent"] == "other_unclear")
        print(f"   non-English/unclear items: {len(ne)}; of those {unc} became other_unclear "
              f"({unc/len(ne):.0%})  -- high share may mean language is being used as a fallback")
    bad = [r["id"] for r in done if r["language"] != "english" and r["escalation"] == "ESCALATE"
           and r.get("escalation_reason") == "out_of_scope"]
    if bad:
        print(f"   CHECK: {bad} escalated non-English as out_of_scope - language alone is not a reason")

    print("\n=== attribute ambiguity ===")
    for col in ("conversation_state", "urgency", "frustration", "language"):
        print(f"   {col:<22} {dict(Counter(r[col] for r in done))}")
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--validate", metavar="CSV")
    ap.add_argument("--csv", default="twcs/twcs.csv")
    ap.add_argument("--brand", default="SpotifyCares")
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="dev (default, avoids tuning against the evaluation set); test only if the owner overrules")
    ap.add_argument("--out", default="outputs/pilot")
    ap.add_argument("--chunksize", type=int, default=400_000)
    args = ap.parse_args()

    if args.validate:
        return validate(args.validate)
    if not args.build:
        ap.error("pass --build or --validate CSV")

    if args.split == "test":
        print("WARNING: sampling the pilot from TEST. Codebook revisions made from these "
              "items tune against the evaluation set (D7). The exclusion file lists ids that "
              "must be kept out of the golden set.", file=sys.stderr)
    m = build(args.csv, args.split, args.out, args.brand, args.chunksize)
    print(json.dumps(m, indent=2))
    print(f"\n[pilot] {m['n_items']} items from {m['split_used']} -> {m['files']['items']}", file=sys.stderr)
    print(f"[peak RSS] {P.peak_mb():.1f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
