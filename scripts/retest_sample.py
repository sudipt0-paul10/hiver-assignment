#!/usr/bin/env python3
"""D12 — delayed blind re-label (test–retest) for intra-annotator reliability.

    python scripts/retest_sample.py --build        # draw the subsample, write the blind artefacts
    python scripts/retest_sample.py --verify       # integrity + blindness + delay gate
    python scripts/retest_sample.py --score data/golden/retest_annotated.csv

WHY THIS EXISTS (D12)
---------------------
The golden set has one annotator and one pass, so there is no human–human
reliability baseline. Without a self-agreement figure, "the judge agrees with the
human at kappa = 0.62" is uninterpretable: it could be a weak judge or an
inherently noisy task. Self-agreement is the **ceiling** that makes judge–human
agreement readable, and deliverable 3 requires judge–human agreement evidence.

HOW BLINDNESS IS PRESERVED
--------------------------
- Items are re-identified R001..R030. The golden id is never shown, so the
  annotator cannot look up the first pass by id.
- The order is reshuffled, so position carries no information.
- Every annotation field is blank; no first-pass label is written to any artefact
  the annotator can open.
- The retest stratum is withheld, exactly as the sampling stratum was.
- The mapping R -> G lives in retest_key.json alone.

Recognition is still possible - it is the same text, and test-retest always has
that limitation. The delay is the mitigation, and it is the reason --build
records an earliest-start date that --verify enforces as a warning.

NOTHING HERE TOUCHES THE 200 ANNOTATIONS. This script only reads them, and
records a SHA-256 of the annotated CSV at build time so any later change is
detectable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate_pilot as AP  # noqa: E402
import profile as P  # noqa: E402
import sample_golden as SG  # noqa: E402

RETEST_SEED = 1212          # D12
EARLIEST_START_DAYS = 5     # the "deliberate delay"; recorded and enforced as a warning

# Quotas. `random` is drawn FIRST and is the only unbiased stratum - the same
# rule the golden sampler follows for its core. The other four deliberately
# over-sample where the first pass showed instability, which is what D12 is for.
N_RANDOM = 12
N_LOW_CONFIDENCE = 7
N_OTHER_UNCLEAR_CONFIDENT = 5
N_BOUNDARY = 3
N_ESCALATION_DEVIANT = 3
# 12 + 7 + 5 + 3 + 3 = 30

SCORED_FIELDS = ["intent", "conversation_state", "urgency", "frustration",
                 "language", "escalation", "escalation_reason", "confidence"]


def load_golden(ann: str, keyfile: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with open(ann, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    key = json.load(open(keyfile, encoding="utf-8"))["items"]
    if set(r["id"] for r in rows) != set(key):
        raise SystemExit("annotation/key id mismatch")
    return rows, key


def majority_escalation(rows: list[dict[str, str]]) -> dict[str, str]:
    by: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by[r["intent"]][r["escalation"]] += 1
    return {k: v.most_common(1)[0][0] for k, v in by.items()}


def build(out_dir: str, ann: str, keyfile: str) -> dict[str, Any]:
    rows, gkey = load_golden(ann, keyfile)
    rng = np.random.default_rng(RETEST_SEED)
    maj = majority_escalation(rows)

    pools = {
        "retest:random": rows,
        "retest:low_confidence": [r for r in rows if r["confidence"] == "low"],
        "retest:other_unclear_confident": [r for r in rows if r["intent"] == "other_unclear"
                                           and r["confidence"] != "low"],
        "retest:boundary": [r for r in rows if gkey[r["id"]]["stratum"].startswith("boundary:")],
        "retest:escalation_deviant": [r for r in rows if r["escalation"] != maj[r["intent"]]],
    }
    quotas = {"retest:random": N_RANDOM,
              "retest:low_confidence": N_LOW_CONFIDENCE,
              "retest:other_unclear_confident": N_OTHER_UNCLEAR_CONFIDENT,
              "retest:boundary": N_BOUNDARY,
              "retest:escalation_deviant": N_ESCALATION_DEVIANT}

    chosen: dict[str, str] = {}
    draw_log = []
    for label, k in quotas.items():          # dict order == draw order; random first
        cand = [r for r in pools[label] if r["id"] not in chosen]
        got = min(k, len(cand))
        draw_log.append({"stratum": label, "requested": k, "eligible_at_draw": len(cand),
                         "drawn": got,
                         "inclusion_prob": round(got / len(cand), 6) if cand else None})
        for i in np.sort(rng.choice(len(cand), size=got, replace=False)):
            chosen[cand[int(i)]["id"]] = label

    sel = [r for r in rows if r["id"] in chosen]
    order = rng.permutation(len(sel))
    sel = [sel[int(i)] for i in order]

    os.makedirs(out_dir, exist_ok=True)
    items = os.path.join(out_dir, "retest_items.csv")
    mapping = {}
    with open(items, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=AP.COLUMNS)
        w.writeheader()
        for i, r in enumerate(sel, start=1):
            rid = f"R{i:03d}"
            mapping[rid] = r["id"]
            w.writerow({"id": rid, "text": r["text"], **{c: "" for c in AP.COLUMNS[2:]}})
    xlsx = SG.write_xlsx(items, os.path.join(out_dir, "retest_items.xlsx"))

    built = datetime.now(timezone.utc)
    key_path = os.path.join(out_dir, "retest_key.json")
    with open(key_path, "w", encoding="utf-8") as fh:
        json.dump({
            "_warning": "RETEST KEY - DO NOT OPEN BEFORE THE SECOND PASS IS COMPLETE. "
                        "It maps each R id back to its golden id; seeing it defeats the "
                        "blindness the whole procedure depends on.",
            "seed": RETEST_SEED,
            "items": {rid: {"golden_id": gid, "retest_stratum": chosen[gid]}
                      for rid, gid in mapping.items()},
        }, fh, indent=2)

    manifest = {
        "procedure": "D12 delayed blind re-label (test-retest), intra-annotator reliability",
        "seed": RETEST_SEED,
        "n_items": len(sel),
        "built_utc": built.isoformat(),
        "earliest_start_utc": (built + timedelta(days=EARLIEST_START_DAYS)).isoformat(),
        "delay_days": EARLIEST_START_DAYS,
        "source_annotations": ann,
        "source_annotations_sha256": SG.sha256_file(ann),
        "source_key_sha256": SG.sha256_file(keyfile),
        "strata": dict(Counter(chosen.values())),
        "draw_log": draw_log,
        "first_pass_labels_in_annotation_artefacts": False,
        "golden_ids_in_annotation_artefacts": False,
        "note": "Only 'retest:random' is unbiased over the 200. The other four strata "
                "deliberately over-sample where the first pass hesitated, so the POOLED "
                "self-agreement is a conservative lower bound, not the population figure. "
                "Report per-stratum agreement, and read the population ceiling off "
                "'retest:random' with its interval.",
        "files": {"items": items, "items_xlsx": xlsx, "key": key_path},
    }
    with open(os.path.join(out_dir, "retest_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def verify(out_dir: str, ann: str) -> int:
    items = os.path.join(out_dir, "retest_items.csv")
    with open(items, encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        fields = list(rdr.fieldnames or [])
        rows = list(rdr)
    key = json.load(open(os.path.join(out_dir, "retest_key.json"), encoding="utf-8"))["items"]
    man = json.load(open(os.path.join(out_dir, "retest_manifest.json"), encoding="utf-8"))
    with open(ann, encoding="utf-8-sig", newline="") as fh:
        golden = {r["id"]: r for r in csv.DictReader(fh)}

    fails: list[str] = []

    def check(ok: bool, name: str, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  - ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    print("=== D12 retest verification ===")
    check(len(rows) == man["n_items"] == 30, "1. row count", f"{len(rows)}")
    ids = [r["id"] for r in rows]
    check(len(set(ids)) == len(ids) and set(ids) == set(key), "2. R ids unique and match the key")
    gids = [key[i]["golden_id"] for i in ids]
    check(len(set(gids)) == len(gids), "3. each R id maps to a distinct golden item",
          f"{len(set(gids))} golden items")
    check(all(g in golden for g in gids), "4. every mapped golden id exists in the 200")
    drift = [r["id"] for r in rows if r["text"] != golden[key[r["id"]]["golden_id"]]["text"]]
    check(not drift, "5. text byte-identical to the golden annotation",
          f"{len(drift)} mismatched" if drift else "30/30")
    check(all(not (r.get(c) or "").strip() for r in rows for c in AP.COLUMNS[2:]),
          "6. every annotation field is blank")
    leak = [r["id"] for r in rows if any(g in " ".join(r.values()) for g in golden)]
    check(fields == AP.COLUMNS and not leak,
          "7. blindness: no golden id and no first-pass label in the annotation CSV")
    check(SG.sha256_file(ann) == man["source_annotations_sha256"],
          "8. the 200 annotations are unchanged since the retest was drawn")
    xl = os.path.join(out_dir, "retest_items.xlsx")
    bad = SG.xlsx_text_mismatches(xl, {r["id"]: r["text"] for r in rows}) if os.path.exists(xl) else ["missing"]
    check(not bad, "9. xlsx companion round-trips the text exactly")

    now = datetime.now(timezone.utc)
    earliest = datetime.fromisoformat(man["earliest_start_utc"])
    ok_delay = now >= earliest
    print(f"  [{'PASS' if ok_delay else 'WAIT'}] 10. delay gate - built {man['built_utc'][:10]}, "
          f"earliest start {man['earliest_start_utc'][:10]} "
          f"({'elapsed' if ok_delay else f'{(earliest - now).days + 1} day(s) to go'})")
    if not ok_delay:
        print("        Not a failure: the artefact is ready, the second pass is not due yet.")

    print("\n=== retest strata ===")
    for s, n in sorted(Counter(v["retest_stratum"] for v in key.values()).items()):
        print(f"  {s:<34} {n:>3}")
    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa. Inlined to avoid a scikit-learn dependency (not installed)."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] / n * cb[c] / n for c in set(ca) | set(cb))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score(path: str, out_dir: str, ann: str) -> int:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        second = {r["id"]: r for r in csv.DictReader(fh)}
    key = json.load(open(os.path.join(out_dir, "retest_key.json"), encoding="utf-8"))["items"]
    with open(ann, encoding="utf-8-sig", newline="") as fh:
        first = {r["id"]: r for r in csv.DictReader(fh)}

    pairs = [(key[rid]["golden_id"], rid, key[rid]["retest_stratum"])
             for rid in key if rid in second and (second[rid].get("intent") or "").strip()]
    if not pairs:
        print("Nothing annotated in the second pass yet. Fill the retest workbook, export it, "
              "then re-run --score.")
        return 1
    print(f"=== D12 intra-annotator reliability  ({len(pairs)}/{len(key)} re-labelled) ===")
    if len(pairs) < len(key):
        print(f"  NOTE: partial - {len(key) - len(pairs)} items not yet re-labelled.")

    print("\n--- per field, all re-labelled items (pooled: CONSERVATIVE, hard-enriched) ---")
    print(f"  {'field':<22}{'agree':>10}{'raw':>9}{'kappa':>9}   Wilson 95%")
    for f in SCORED_FIELDS:
        a = [first[g][f].strip() for g, r, _ in pairs]
        b = [second[r][f].strip() for g, r, _ in pairs]
        k = sum(1 for x, y in zip(a, b) if x == y)
        lo, hi = P.wilson(k, len(a))
        print(f"  {f:<22}{k:>5}/{len(a):<4}{k/len(a):>8.1%}{kappa(a, b):>9.3f}   "
              f"[{lo:.1%}, {hi:.1%}]")

    print("\n--- 'retest:random' only: the unbiased self-agreement ceiling ---")
    rnd = [p for p in pairs if p[2] == "retest:random"]
    if rnd:
        for f in ("intent", "escalation"):
            a = [first[g][f].strip() for g, r, _ in rnd]
            b = [second[r][f].strip() for g, r, _ in rnd]
            k = sum(1 for x, y in zip(a, b) if x == y)
            lo, hi = P.wilson(k, len(a))
            print(f"  {f:<22}{k:>5}/{len(a):<4}{k/len(a):>8.1%}   [{lo:.1%}, {hi:.1%}]")
        print("  n is small by design; quote the interval, never the point estimate alone.")

    print("\n--- per stratum (intent / escalation raw agreement) ---")
    for s in sorted({p[2] for p in pairs}):
        grp = [p for p in pairs if p[2] == s]
        ki = sum(1 for g, r, _ in grp if first[g]["intent"].strip() == second[r]["intent"].strip())
        ke = sum(1 for g, r, _ in grp
                 if first[g]["escalation"].strip() == second[r]["escalation"].strip())
        print(f"  {s:<34} intent {ki}/{len(grp)}   escalation {ke}/{len(grp)}")

    print("\n--- disagreements (first pass -> second pass) ---")
    for g, r, s in sorted(pairs, key=lambda p: p[2]):
        diffs = [f"{f}: {first[g][f].strip()!r} -> {second[r][f].strip()!r}"
                 for f in SCORED_FIELDS if first[g][f].strip() != second[r][f].strip()]
        if diffs:
            print(f"  {g} (as {r}) [{s}]")
            print(f"    {first[g]['text'][:96]}")
            for d in diffs:
                print(f"    - {d}")
    print("\nDisagreements are a reliability measurement. Do NOT edit either pass to "
          "reconcile them:\nthe golden set's first pass stays authoritative.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--score", metavar="CSV")
    ap.add_argument("--out", default="data/golden")
    ap.add_argument("--annotations", default="data/golden/golden_annotated.csv")
    ap.add_argument("--key", default="data/golden/golden_key.json")
    args = ap.parse_args()

    if args.score:
        return score(args.score, args.out, args.annotations)
    if args.verify:
        return verify(args.out, args.annotations)
    if not args.build:
        ap.error("pass --build, --verify or --score CSV")
    m = build(args.out, args.annotations, args.key)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
