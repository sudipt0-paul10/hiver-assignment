#!/usr/bin/env python3
"""Characterise the completed 200-item golden set. No model, no API, no labels written.

    python scripts/golden_report.py                      # text report to stdout
    python scripts/golden_report.py --json outputs/golden/golden_report.json

WHAT THIS IS NOT
----------------
This is **not** a system evaluation. No agent, classifier, retriever or judge
exists yet, so there are no predictions to score and no accuracy, F1 or
prediction-vs-gold confusion matrix to compute. `golden_key.json` holds sampling
metadata - stratum, ids, inclusion probability, design weight - and contains no
reference labels, so there is nothing to score the human annotations against
either.

What it does produce is everything the labels themselves support: distributions
with Wilson intervals, core vs targeted vs full, the escalation policy's shape,
the crude seed-proxy-vs-label confusion, and the ambiguity register that decides
what has to be resolved before a system is built on top of this set.

Annotations are read-only here. Two rows carry an intent outside the locked
taxonomy; they are reported as INVALID in every table and never reassigned.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate_pilot as AP  # noqa: E402
import profile as P  # noqa: E402

UNCLEAR_THRESHOLD = 0.15   # docs/taxonomy.md 7: above this the taxonomy is inadequate
NEAR_CONSTANT = 0.90       # an attribute this concentrated carries little information


def ci(k: int, n: int) -> str:
    lo, hi = P.wilson(k, n)
    return f"{k/n:6.1%}  [{lo:.1%}, {hi:.1%}]" if n else "   n/a"


def load(ann: str, keyfile: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with open(ann, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    key = json.load(open(keyfile, encoding="utf-8"))["items"]
    missing = sorted(set(r["id"] for r in rows) ^ set(key))
    if missing:
        raise SystemExit(f"annotation/key id mismatch: {missing[:10]}")
    return rows, key


def subsets(rows: list[dict[str, str]], key: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    core = [r for r in rows if key[r["id"]]["stratum"] == "core"]
    targ = [r for r in rows if key[r["id"]]["stratum"] != "core"]
    return {"CORE (unbiased)": core, "TARGETED (enriched)": targ, "FULL": rows}


def dist_block(title: str, col: str, sets: dict[str, list[dict[str, str]]],
               order: list[str], invalid_label: str = "INVALID (outside taxonomy)") -> None:
    print(f"\n--- {title} ---")
    seen = {v for grp in sets.values() for r in grp for v in [r[col].strip()]}
    extra = sorted(seen - set(order) - {""})
    print(f"  {'value':<30}" + "".join(f"{n:>28}" for n in sets))
    for v in order + extra:
        cells = []
        for grp in sets.values():
            k = sum(1 for r in grp if r[col].strip() == v)
            cells.append(f"{k:>4}  {ci(k, len(grp)):>21}" if grp else "")
        tag = f"{v}  <- {invalid_label}" if v in extra else v
        print(f"  {tag:<30}" + "".join(f"{c:>28}" for c in cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default="data/golden/golden_annotated.csv")
    ap.add_argument("--key", default="data/golden/golden_key.json")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    rows, key = load(args.annotations, args.key)
    sets = subsets(rows, key)
    core, targ = sets["CORE (unbiased)"], sets["TARGETED (enriched)"]
    out: dict[str, Any] = {"n_items": len(rows), "n_core": len(core), "n_targeted": len(targ)}

    print("=" * 96)
    print("GOLDEN SET CHARACTERISATION - human labels only, no system exists yet")
    print("=" * 96)
    print(f"items {len(rows)}   core {len(core)}   targeted {len(targ)}")
    print("Intervals are Wilson 95%. Only CORE supports a population statement;")
    print("every core item shares one inclusion probability, so its design-weighted")
    print("and unweighted estimates are identical.")

    # ---- 1. schema violations -------------------------------------------------
    invalid = [r for r in rows if r["intent"].strip() not in AP.INTENTS]
    print(f"\n[1] SCHEMA VIOLATIONS: {len(invalid)}")
    for r in invalid:
        print(f"    {r['id']}  intent='{r['intent']}'  stratum={key[r['id']]['stratum']}")
        print(f"        {r['text'][:110]}")
    out["schema_violations"] = [{"id": r["id"], "intent": r["intent"]} for r in invalid]

    # ---- 2. intent ------------------------------------------------------------
    dist_block("[2] INTENT", "intent", sets, AP.INTENTS)
    nu = sum(1 for r in core if r["intent"] == "other_unclear")
    nu_f = sum(1 for r in rows if r["intent"] == "other_unclear")
    inv_c = sum(1 for r in core if r in invalid)
    print(f"\n  other_unclear vs the {UNCLEAR_THRESHOLD:.0%} falsification threshold "
          f"(docs/taxonomy.md 7):")
    print(f"    as annotated        core {ci(nu, len(core))}   full {ci(nu_f, len(rows))}")
    print(f"    if the {len(invalid)} INVALID rows resolved to other_unclear   "
          f"core {ci(nu + inv_c, len(core))}   full {ci(nu_f + len(invalid), len(rows))}")
    print("    -> the verdict on the locked taxonomy depends on how those rows are ruled.")
    print("       Reported, not resolved: no annotation was changed.")
    out["other_unclear"] = {"core": nu, "full": nu_f, "core_n": len(core),
                            "threshold": UNCLEAR_THRESHOLD,
                            "if_invalid_resolved_core": nu + inv_c,
                            "if_invalid_resolved_full": nu_f + len(invalid)}

    # ---- 3. attributes --------------------------------------------------------
    for title, col, order in (("[3a] CONVERSATION STATE", "conversation_state", AP.CONVERSATION_STATE),
                              ("[3b] URGENCY", "urgency", AP.LEVELS),
                              ("[3c] FRUSTRATION", "frustration", AP.LEVELS),
                              ("[3d] LANGUAGE", "language", AP.LANGUAGE),
                              ("[3e] CONFIDENCE", "confidence", AP.CONFIDENCE)):
        dist_block(title, col, sets, order)
    print("\n  informativeness - an attribute concentrated above "
          f"{NEAR_CONSTANT:.0%} in one value is close to constant and will not")
    print("  discriminate at evaluation time:")
    for col in ("conversation_state", "urgency", "frustration", "language"):
        c = Counter(r[col] for r in rows).most_common(1)[0]
        flag = "  <- NEAR-CONSTANT" if c[1] / len(rows) >= NEAR_CONSTANT else ""
        print(f"    {col:<22} dominant '{c[0]}' {c[1]/len(rows):.1%}{flag}")
    out["attributes"] = {c: dict(Counter(r[c] for r in rows))
                         for c in ("conversation_state", "urgency", "frustration",
                                   "language", "confidence")}

    # ---- 4. escalation --------------------------------------------------------
    print("\n--- [4] ESCALATION ---")
    for name, grp in sets.items():
        k = sum(1 for r in grp if r["escalation"] == "ESCALATE")
        print(f"  {name:<22} ESCALATE {ci(k, len(grp))}")
    print("\n  The TARGETED rate is inflated by construction - those 40 items were drawn")
    print("  from escalation and boundary proxies. Only the CORE rate is a base rate.")
    out["escalation_rate"] = {n: sum(1 for r in g if r["escalation"] == "ESCALATE")
                              for n, g in sets.items()}

    print("\n  reason distribution (ESCALATE rows only):")
    for name, grp in sets.items():
        rc = Counter(r["escalation_reason"] for r in grp if r["escalation"] == "ESCALATE")
        print(f"    {name}:")
        for reason in AP.REASONS[1:]:
            n = rc.get(reason, 0)
            mark = "  <- EMPTY CELL" if n == 0 and name == "FULL" else ""
            print(f"      {reason:<22} {n:>3}{mark}")
    out["escalation_reasons"] = dict(Counter(
        r["escalation_reason"] for r in rows if r["escalation"] == "ESCALATE"))

    # ---- 5. is escalation just a function of intent? --------------------------
    print("\n--- [5] DOES INTENT ALONE DETERMINE ESCALATION? ---")
    by = defaultdict(Counter)
    for r in rows:
        by[r["intent"]][r["escalation"]] += 1
    hits = 0
    for intent in sorted(by):
        c = by[intent]
        top = c.most_common(1)[0]
        hits += top[1]
        pure = "PURE" if len(c) == 1 else "mixed"
        print(f"    {intent:<26} {dict(c)}  {pure}")
    print(f"\n  A lookup table mapping each intent to its majority escalation decision")
    print(f"  reproduces {hits}/{len(rows)} = {hits/len(rows):.1%} of the human decisions.")
    print("  This is an IN-SAMPLE ceiling fitted on these very labels, so it overstates")
    print("  what such a rule would score out of sample - but it is the baseline the")
    print("  escalation headline must be reported against. Without it, an escalation")
    print("  classifier that has merely learned the intent looks like a safety result.")
    out["intent_majority_escalation_insample"] = {"correct": hits, "n": len(rows)}

    # ---- 6. seed proxy vs human label ----------------------------------------
    print("\n--- [6] SEED PROXY vs HUMAN LABEL (strata are retrieval devices, not labels) ---")
    print("  Disagreement is a boundary signal, never an annotation error.")
    agree = total = 0
    for r in sorted(rows, key=lambda x: key[x["id"]]["stratum"]):
        st = key[r["id"]]["stratum"]
        if not st.startswith("targeted:"):
            continue
        exp = st.split(":", 1)[1]
        total += 1
        agree += exp == r["intent"]
        if exp != r["intent"]:
            print(f"    {r['id']} seeded {exp:<26} -> labelled {r['intent']:<26} "
                  f"conf={r['confidence']}")
    print(f"    targeted agreement {agree}/{total}")

    print("\n  boundary pairs - did the human land inside the pair the proxy predicted?")
    b_in = b_tot = 0
    for r in rows:
        st = key[r["id"]]["stratum"]
        if not st.startswith("boundary:"):
            continue
        pair = st.split(":", 1)[1].split("|")
        b_tot += 1
        inside = r["intent"] in pair
        b_in += inside
        print(f"    {r['id']} {'/'.join(pair):<50} -> {r['intent']:<26} "
              f"{'in pair' if inside else 'OUTSIDE PAIR'}  conf={r['confidence']}")
    print(f"    landed inside the seeded pair: {b_in}/{b_tot}")

    print("\n  escalation proxies - did the enriched item actually escalate?")
    for st in sorted({key[r["id"]]["stratum"] for r in rows
                      if key[r["id"]]["stratum"].startswith("escalation:")}):
        grp = [r for r in rows if key[r["id"]]["stratum"] == st]
        k = sum(1 for r in grp if r["escalation"] == "ESCALATE")
        reasons = Counter(r["escalation_reason"] for r in grp if r["escalation"] == "ESCALATE")
        want = st.split(":", 1)[1]
        on_target = reasons.get(want, 0)
        print(f"    {st:<40} escalated {k}/{len(grp)}  reason matched proxy {on_target}/{len(grp)}"
              f"  {dict(reasons)}")
    out["proxy_agreement"] = {"targeted": [agree, total], "boundary_in_pair": [b_in, b_tot]}

    print("\n  vague_unclear stratum:")
    for r in rows:
        if key[r["id"]]["stratum"] == "vague_unclear":
            print(f"    {r['id']} -> {r['intent']:<26} {r['escalation']:<9} conf={r['confidence']}")

    # ---- 7. leakage / isolation ----------------------------------------------
    print("\n--- [7] customer_also_in_train (retained by design; a retrieval-time constraint) ---")
    tr = [r for r in rows if key[r["id"]]["customer_also_in_train"]]
    for r in tr:
        print(f"    {r['id']}  customer={key[r['id']]['customer_id']:<10} "
              f"stratum={key[r['id']]['stratum']:<12} intent={r['intent']}")
    print(f"    {len(tr)} items. They stay in the set - dropping them would bias it against")
    print("    repeat customers. Retrieval MUST exclude same-customer and same-thread")
    print("    evidence for these ids (docs/golden_set.md 2).")
    out["customer_also_in_train"] = [r["id"] for r in tr]

    # ---- 8. ambiguity register ------------------------------------------------
    print("\n--- [8] AMBIGUITY REGISTER - what must be resolved before a system is built ---")
    low = [r for r in rows if r["confidence"] == "low"]
    print(f"  low confidence: {len(low)} ({len(low)/len(rows):.1%})")
    for r in low:
        note = r["notes"].strip() or "*** NO NOTE ***"
        print(f"    {r['id']} [{r['intent']}] {note}")
    cr = [r for r in rows if r["cannot_represent"].strip()]
    print(f"\n  cannot_represent: {len(cr)}")
    if not cr and invalid:
        print("    NOTE: zero uses, yet two items were given an invented intent instead.")
        print("    The codebook's escape hatch for an unrepresentable item was not exercised.")
    acct = {"account_access", "security"}
    flag = [r for r in rows if r["escalation_reason"].strip() in acct
            and r["intent"] != "account_access"]
    print(f"\n  coherence flags (account-family reason, non-account intent): {len(flag)} - advisory")
    for r in flag:
        print(f"    {r['id']} intent={r['intent']:<24} reason={r['escalation_reason']:<16} "
              f"{r['text'][:56]}")
    print(f"\n  notes recorded: {sum(1 for r in rows if r['notes'].strip())}/{len(rows)}")
    out["ambiguity"] = {"low_confidence": [r["id"] for r in low],
                        "cannot_represent": [r["id"] for r in cr],
                        "coherence_flags": [r["id"] for r in flag],
                        "notes_recorded": sum(1 for r in rows if r["notes"].strip())}

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[json] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
