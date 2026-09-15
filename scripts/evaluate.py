#!/usr/bin/env python3
"""Evaluation harness: D3 headline, D5 risk-coverage, D7 intervals and McNemar.

    python scripts/evaluate.py --intent
    python scripts/evaluate.py --escalation
    python scripts/evaluate.py --all --json outputs/eval/results.json

Reads prediction files from outputs/preds/ and the human labels from
data/golden/golden_annotated.csv. This is the ONLY module that opens the golden
labels; every classifier and policy is built and run without them (D23).

DEFINITIONS, fixed by D3 and not restated loosely anywhere else:

    coverage         = share of evaluated items the system chooses to auto-handle
    unsafe-auto rate = of the auto-handled items, the share the HUMAN labelled
                       ESCALATE
                       (denominator = auto-handled, numerator = those whose human
                        label is ESCALATE)

Unsafe-auto is meaningless without coverage - a system that auto-handles nothing
scores 0%. Systems are therefore compared at MATCHED COVERAGE, and the full curve
is reported.

NO SUPERVISED METRIC APPEARS HERE. There is no human-labelled training set (D23),
so every intent number below is the score of a rule, a distant-supervision proxy,
or an LLM prompted with the codebook. Each is printed with its construction.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import Counter
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate_pilot as AP  # noqa: E402
import profile as P  # noqa: E402

PRED_DIR = os.path.join("outputs", "preds")


def ci(k: int, n: int) -> str:
    if n == 0:
        return "     n/a"
    lo, hi = P.wilson(k, n)
    return f"{k/n:6.1%} [{lo:.1%}, {hi:.1%}]"


def load_gold() -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    with open("data/golden/golden_annotated.csv", encoding="utf-8-sig", newline="") as fh:
        gold = pd.DataFrame(list(csv.DictReader(fh)))
    key = json.load(open("data/golden/golden_key.json", encoding="utf-8"))["items"]
    subset = {gid: ("core" if v["stratum"] == "core" else "targeted") for gid, v in key.items()}
    family = {gid: v["stratum"].split(":", 1)[0] for gid, v in key.items()}
    return gold, subset, family


def meta_for(path: str) -> dict[str, Any]:
    m = path.replace(".csv", ".meta.json")
    return json.load(open(m, encoding="utf-8")) if os.path.exists(m) else {}


# ---------------------------------------------------------------------------
# intent
# ---------------------------------------------------------------------------
def prf(gold: list[str], pred: list[str], labels: list[str]) -> dict[str, Any]:
    out, f1s = {}, []
    for c in labels:
        tp = sum(1 for g, p_ in zip(gold, pred) if g == c and p_ == c)
        fp = sum(1 for g, p_ in zip(gold, pred) if g != c and p_ == c)
        fn = sum(1 for g, p_ in zip(gold, pred) if g == c and p_ != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"support": tp + fn, "precision": prec, "recall": rec, "f1": f1}
        if tp + fn:                      # macro over classes actually present in gold
            f1s.append(f1)
    out["_macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    return out


def eval_intent(gold: pd.DataFrame, subset: dict[str, str],
                family: dict[str, str]) -> dict[str, Any]:
    g = dict(zip(gold["id"], gold["intent"]))
    invalid = [i for i, v in g.items() if v not in AP.INTENTS]
    print("=" * 92)
    print("INTENT  — no supervised baseline exists (D23); read every row with its construction")
    print("=" * 92)
    print(f"{len(invalid)} gold item(s) carry an intent outside the taxonomy "
          f"({', '.join(invalid)}); no classifier can emit those values, so they are "
          f"counted as errors.\nAccuracy excluding them is shown alongside.")

    results: dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(PRED_DIR, "intent_*_golden.csv"))):
        name = os.path.basename(path).replace("intent_", "").replace("_golden.csv", "")
        meta = meta_for(path)
        pred = dict(zip(*pd.read_csv(path)[["id", "pred_intent"]].values.T))
        ids = [i for i in gold["id"] if i in pred]
        rows = {"full": ids,
                "core": [i for i in ids if subset[i] == "core"],
                "targeted": [i for i in ids if subset[i] == "targeted"],
                "valid-gold-only": [i for i in ids if i not in invalid]}
        print(f"\n--- {name} ---")
        print(f"    construction: {meta.get('construction', 'unrecorded')}")
        if meta.get("proxy_baseline"):
            print("    LABEL TYPE:   distant supervision (proxy baseline) — NOT supervised")
        res: dict[str, Any] = {"construction": meta.get("construction"),
                               "proxy_baseline": bool(meta.get("proxy_baseline"))}
        for sub, sids in rows.items():
            k = sum(1 for i in sids if g[i] == pred[i])
            print(f"    accuracy {sub:<16} {ci(k, len(sids))}   n={len(sids)}")
            res[f"accuracy_{sub}"] = {"correct": k, "n": len(sids)}
        det = prf([g[i] for i in ids], [pred[i] for i in ids], AP.INTENTS)
        res["macro_f1_full"] = det["_macro_f1"]
        print(f"    macro-F1 (8 classes, full)  {det['_macro_f1']:.3f}")
        print(f"    {'class':<26}{'support':>8}{'prec':>8}{'rec':>8}{'F1':>8}")
        for c in AP.INTENTS:
            d = det[c]
            print(f"    {c:<26}{d['support']:>8}{d['precision']:>8.2f}"
                  f"{d['recall']:>8.2f}{d['f1']:>8.2f}")
        res["per_class"] = {c: det[c] for c in AP.INTENTS}
        conf = Counter((g[i], pred[i]) for i in ids if g[i] != pred[i])
        print("    top confusions (gold -> predicted):")
        for (a, b), n in conf.most_common(6):
            print(f"      {a:<26} -> {b:<26} {n:>3}")
        res["top_confusions"] = [{"gold": a, "pred": b, "n": n} for (a, b), n in conf.most_common(10)]

        print("    accuracy by sampling-stratum family (see the circularity warning below):")
        fam_acc = {}
        for fam in sorted(set(family[i] for i in ids)):
            fids = [i for i in ids if family[i] == fam]
            k = sum(1 for i in fids if g[i] == pred[i])
            mark = "   <- CIRCULAR: drawn because a seed matched" if fam == "targeted" else ""
            print(f"      {fam:<18} {ci(k, len(fids))}   n={len(fids)}{mark}")
            fam_acc[fam] = {"correct": k, "n": len(fids)}
        res["accuracy_by_stratum_family"] = fam_acc
        results[name] = res
        results.setdefault("_correct", {})[name] = [g[i] == pred[i] for i in gold["id"] if i in pred]

    print("\n--- CIRCULARITY WARNING ---")
    print("  The 14 `targeted:<intent>` items were SELECTED because a seed family matched")
    print("  them. The rule classifier is those same seeds, and distant-lr is fitted on their")
    print("  output, so both are being scored partly on items chosen for agreeing with them.")
    print("  Accuracy on the `targeted` family is inflated by construction and must never be")
    print("  quoted as a headline. The 160-item core carries no such circularity.")

    correct = results.pop("_correct", {})
    names = sorted(correct)
    if len(names) > 1:
        print("\n--- paired McNemar, exact two-sided, on intent correctness (D7) ---")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                m = mcnemar_exact(correct[a], correct[b])
                star = "  *" if m["p_value"] < 0.05 else ""
                print(f"  {a:<24} vs {b:<24} b={m['b']:>3} c={m['c']:>3} "
                      f"p={m['p_value']:.4f}{star}")
                results.setdefault("_mcnemar", {})[f"{a}|{b}"] = m
    return results


# ---------------------------------------------------------------------------
# escalation
# ---------------------------------------------------------------------------
def unsafe_auto(gold_esc: dict[str, str], decision: dict[str, str],
                ids: list[str]) -> tuple[int, int, int]:
    """(unsafe, auto_handled, n) — the D3 numerator, denominator and universe."""
    auto = [i for i in ids if decision[i] == "AUTO_OK"]
    unsafe = sum(1 for i in auto if gold_esc[i] == "ESCALATE")
    return unsafe, len(auto), len(ids)


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict[str, Any]:
    """Exact two-sided binomial McNemar on paired correctness. No scipy dependency."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "note": "no discordant pairs"}
    p = 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return {"b": b, "c": c, "p_value": min(1.0, p)}


def risk_coverage(gold_esc: dict[str, str], score: dict[str, float],
                  ids: list[str]) -> list[dict[str, float]]:
    """Sweep the auto-handle threshold: auto-handle when score < t."""
    pts = []
    for t in sorted({0.0} | {score[i] for i in ids} | {1.0001}):
        auto = [i for i in ids if score[i] < t]
        uns = sum(1 for i in auto if gold_esc[i] == "ESCALATE")
        pts.append({"threshold": round(t, 6), "coverage": len(auto) / len(ids),
                    "unsafe_auto": (uns / len(auto)) if auto else 0.0,
                    "n_auto": len(auto), "n_unsafe": uns})
    return pts


def eval_escalation(gold: pd.DataFrame, subset: dict[str, str]) -> dict[str, Any]:
    gesc = dict(zip(gold["id"], gold["escalation"]))
    print("\n" + "=" * 92)
    print("ESCALATION — D3 headline: unsafe-auto rate at matched coverage")
    print("=" * 92)
    base = sum(1 for v in gesc.values() if v == "ESCALATE")
    print(f"human ESCALATE base rate (full 200): {ci(base, len(gesc))}")
    core_ids = [i for i in gold['id'] if subset[i] == 'core']
    print(f"human ESCALATE base rate (core 160): "
          f"{ci(sum(1 for i in core_ids if gesc[i] == 'ESCALATE'), len(core_ids))}"
          "   <- the only population base rate")

    loaded: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(PRED_DIR, "escalation_*_golden.csv"))):
        name = os.path.basename(path).replace("escalation_", "").replace("_golden.csv", "")
        df = pd.read_csv(path, keep_default_na=False)
        loaded[name] = {"decision": dict(zip(df["id"], df["decision"])),
                        "score": dict(zip(df["id"], df["score"].astype(float))),
                        "reason": dict(zip(df["id"], df["reason"]))
                        if "reason" in df.columns else {},
                        "meta": meta_for(path)}

    results: dict[str, Any] = {}
    print(f"\n{'policy':<30}{'coverage':>10}{'unsafe-auto (Wilson 95%)':>30}  labels?")
    for name, d in loaded.items():
        ids = [i for i in gold["id"] if i in d["decision"]]
        uns, auto, n = unsafe_auto(gesc, d["decision"], ids)
        tag = "USES GOLD" if d["meta"].get("uses_human_labels") else "clean"
        print(f"{name:<30}{auto/n:>9.1%}{ci(uns, auto):>30}  {tag}")
        results[name] = {"coverage": auto / n, "unsafe": uns, "auto": auto, "n": n,
                         "construction": d["meta"].get("construction"),
                         "uses_human_labels": bool(d["meta"].get("uses_human_labels"))}
    print("\n  always-escalate and never-escalate are the two endpoints D5 requires:")
    print("  coverage 0% is trivially safe and useless; coverage 100% scores the base rate.")

    # --- risk-coverage curve for the only scored policy --------------------
    scored = {k: v for k, v in loaded.items() if len(set(v["score"].values())) > 2}
    for name, d in scored.items():
        ids = [i for i in gold["id"] if i in d["score"]]
        curve = risk_coverage(gesc, d["score"], ids)
        results[name]["risk_coverage"] = curve
        print(f"\n--- risk–coverage curve: {name} (auto-handle when score < t) ---")
        print(f"  {'coverage':>10}{'unsafe-auto':>14}{'n_auto':>9}{'n_unsafe':>10}")
        seen = set()
        for target in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            pt = min(curve, key=lambda p: abs(p["coverage"] - target))
            if pt["threshold"] in seen:
                continue
            seen.add(pt["threshold"])
            print(f"  {pt['coverage']:>9.1%}{pt['unsafe_auto']:>14.1%}"
                  f"{pt['n_auto']:>9}{pt['n_unsafe']:>10}")

        print(f"\n--- matched-coverage comparison against {name} ---")
        for other, od in loaded.items():
            if other == name or other in scored:
                continue
            oids = [i for i in gold["id"] if i in od["decision"]]
            o_uns, o_auto, o_n = unsafe_auto(gesc, od["decision"], oids)
            cov = o_auto / o_n
            pt = min(curve, key=lambda p: abs(p["coverage"] - cov))
            flag = "  (uses gold labels)" if od["meta"].get("uses_human_labels") else ""
            print(f"  vs {other:<28} coverage {cov:>6.1%}: {other} {o_uns}/{o_auto} "
                  f"= {(o_uns/o_auto if o_auto else 0):.1%}   |   {name} at coverage "
                  f"{pt['coverage']:.1%}: {pt['n_unsafe']}/{pt['n_auto']} "
                  f"= {pt['unsafe_auto']:.1%}{flag}")

    # --- paired McNemar on escalation correctness --------------------------
    print("\n--- paired McNemar, exact two-sided, on escalation correctness (D7) ---")
    names = list(loaded)
    ids = list(gold["id"])
    correct = {n: [loaded[n]["decision"][i] == gesc[i] for i in ids] for n in names}
    for n in names:
        k = sum(correct[n])
        print(f"  {n:<30} accuracy {ci(k, len(ids))}")
    print()
    pairs_out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if {a, b} & {"always-escalate", "never-escalate"}:
                continue
            m = mcnemar_exact(correct[a], correct[b])
            star = "  *" if m["p_value"] < 0.05 else ""
            print(f"  {a:<30} vs {b:<30} b={m['b']:>3} c={m['c']:>3} "
                  f"p={m['p_value']:.4f}{star}")
            pairs_out[f"{a}|{b}"] = m
    print("\n  b = first correct where second wrong; c = the reverse. * = p < 0.05.")
    print("  Constant policies are excluded from pairing: comparing anything to a constant")
    print("  measures the base rate, not the systems.")
    results["_mcnemar"] = pairs_out
    return results


# ---------------------------------------------------------------------------
# reason codes + handoff
# ---------------------------------------------------------------------------
def eval_reasons(gold: pd.DataFrame) -> dict[str, Any]:
    import handoff as H
    gesc = dict(zip(gold["id"], gold["escalation"]))
    greason = dict(zip(gold["id"], gold["escalation_reason"]))
    valid = set(AP.REASONS[1:])

    print("\n" + "=" * 92)
    print("ESCALATION REASON CODES (D9) — deterministic rules, nothing tuned on gold")
    print("=" * 92)

    results: dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(PRED_DIR, "escalation_*_golden.csv"))):
        name = os.path.basename(path).replace("escalation_", "").replace("_golden.csv", "")
        df = pd.read_csv(path, keep_default_na=False)
        if "reason" not in df.columns:
            continue
        meta = meta_for(path)
        esc = df[df["decision"] == "ESCALATE"]
        auto = df[df["decision"] == "AUTO_OK"]

        # --- structural validity: the two invariants the task asks for ---
        missing = esc[esc["reason"].str.strip() == ""]
        illegal = esc[~esc["reason"].isin(valid)]
        multi = esc[esc["reason"].str.contains(r"[,;|]", regex=True)]
        stray = auto[auto["reason"].str.strip() != ""]
        ok = not len(missing) and not len(illegal) and not len(multi) and not len(stray)
        print(f"\n--- {name} ---")
        print(f"  [{'PASS' if ok else 'FAIL'}] every ESCALATE carries exactly one valid "
              f"reason  ({len(esc)} escalated: {len(missing)} missing, {len(illegal)} "
              f"illegal, {len(multi)} multi-valued)")
        print(f"  [{'PASS' if not len(stray) else 'FAIL'}] every AUTO_OK carries no reason  "
              f"({len(auto)} auto-handled, {len(stray)} with a stray reason)")

        res: dict[str, Any] = {"n_escalated": len(esc), "valid": bool(ok),
                               "rule_activations": meta.get("reason_stage", {})
                               .get("rule_activations", {})}

        # --- rule coverage: how often did a text rule fire vs the fallback? ---
        acts = res["rule_activations"]
        by_rule = sum(v for k, v in acts.items() if k.startswith("rule:"))
        by_fb = sum(v for k, v in acts.items() if k.startswith("fallback:"))
        tot = by_rule + by_fb
        if tot:
            print(f"  rule coverage: {by_rule}/{tot} = {by_rule/tot:.1%} decided by a text "
                  f"rule, {by_fb} by the intent fallback")
        res["rule_coverage"] = {"by_rule": by_rule, "by_fallback": by_fb}

        # --- agreement with the human reason, on items BOTH escalate ---
        both = [i for i in df["id"] if gesc.get(i) == "ESCALATE"
                and dict(zip(df["id"], df["reason"]))[i]]
        if both:
            pr = dict(zip(df["id"], df["reason"]))
            k = sum(1 for i in both if pr[i] == greason[i])
            print(f"  reason agreement where BOTH escalate: {ci(k, len(both))}  n={len(both)}")
            conf = Counter((greason[i], pr[i]) for i in both if pr[i] != greason[i])
            for (a, b), n in conf.most_common(5):
                print(f"      human {a:<22} -> predicted {b:<22} {n:>3}")
            res["reason_agreement"] = {"correct": k, "n": len(both)}
            res["top_reason_confusions"] = [{"human": a, "pred": b, "n": n}
                                            for (a, b), n in conf.most_common(10)]
        results[name] = res

    print("\n  Reason agreement is an EVALUATION result, not a tuning signal: the rules")
    print("  were written from D9 and the codebook before this comparison was run, and")
    print("  nothing was changed after seeing it.")

    # --- coverage of the human reason vocabulary -------------------------
    print("\n--- reason codes the rules can never produce on this data ---")
    hum = Counter(greason[i] for i in gold["id"] if gesc[i] == "ESCALATE")
    gaps = []
    for r in AP.REASONS[1:]:
        fired = any(isinstance(v, dict)
                    and (v.get("rule_activations") or {}).get(f"rule:{r}", 0) > 0
                    for v in results.values())
        if hum.get(r, 0) and not fired:
            gaps.append(r)
            print(f"  {r:<22} human used it {hum[r]:>3}x, no rule ever fired  <- COVERAGE GAP")
    if not gaps:
        print("  none - every reason the human used is reachable by at least one rule")
    results["_reason_rule_coverage_gaps"] = gaps
    results["_human_reason_distribution"] = dict(hum)
    return results


def eval_handoff(gold: pd.DataFrame) -> dict[str, Any]:
    import handoff as H
    print("\n" + "=" * 92)
    print("DETERMINISTIC HANDOFF (D8) — no generation on the escalation path")
    print("=" * 92)
    gtext = dict(zip(gold["id"], gold["text"]))
    out: dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(PRED_DIR, "handoff_*_golden.csv"))):
        name = os.path.basename(path).replace("handoff_", "").replace("_golden.csv", "")
        df = pd.read_csv(path, keep_default_na=False)
        bad = df[~df["guardrails_passed"].astype(bool)]
        distinct = df["reply"].nunique()
        echo = [r["id"] for _, r in df.iterrows()
                if not H.guardrails(r["reply"], gtext.get(r["id"], "")).get(
                    "no_customer_text_echoed", True)]
        print(f"\n--- {name} ---")
        print(f"  [{'PASS' if not len(bad) else 'FAIL'}] all {len(df)} handoff replies pass "
              f"every guardrail  ({len(bad)} failures)")
        print(f"  [{'PASS' if not echo else 'FAIL'}] no reply echoes any 6-word window of the "
              f"customer's message  ({len(echo)} violations)")
        print(f"  [{'PASS' if distinct <= len(H.TEMPLATES) else 'FAIL'}] output space bounded "
              f"by the template set: {distinct} distinct replies <= {len(H.TEMPLATES)} templates")
        out[name] = {"n": len(df), "guardrail_failures": int(len(bad)),
                     "distinct_replies": int(distinct), "echo_violations": len(echo)}
    if not out:
        print("\n  no handoff files yet - run scripts/handoff.py --render POLICY")
    return out


def eval_drafts(gold: pd.DataFrame) -> dict[str, Any]:
    """Route mix, guardrail outcomes and judge verdicts for the generation path."""
    gesc = dict(zip(gold["id"], gold["escalation"]))
    print("\n" + "=" * 92)
    print("DRAFTING + JUDGE (D8 generation path, D11 judge)")
    print("=" * 92)
    out: dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(PRED_DIR, "drafts_*_golden*.csv"))):
        name = os.path.basename(path).replace("drafts_", "").replace(".csv", "")
        df = pd.read_csv(path, keep_default_na=False)
        dry = name.endswith("_dryrun")
        routes = Counter(df["route"])
        auto = df[df["route"] == "auto_reply"]
        unsafe = sum(1 for _, r in auto.iterrows() if gesc.get(r["id"]) == "ESCALATE")
        print(f"\n--- {name}{'  [DRY RUN - offline stub, not model output]' if dry else ''} ---")
        print(f"  routes: {dict(routes)}")
        print(f"  guardrail fallbacks to handoff: {routes.get('handoff_fallback', 0)}")
        print(f"  unsafe-auto among generated replies: {ci(unsafe, len(auto))}  n={len(auto)}")
        out[name] = {"routes": dict(routes), "unsafe_auto": unsafe, "n_auto": len(auto),
                     "dry_run": dry}
        jp = path.replace("drafts_", "judge_")
        if os.path.exists(jp):
            j = pd.read_csv(jp, keep_default_na=False)
            ok = j[j["parsed"] == True]  # noqa: E712
            acc = sum(1 for _, r in ok.iterrows() if str(r["acceptable"]).lower() == "true")
            print(f"  judge: {len(ok)}/{len(j)} parsed, acceptable {ci(acc, len(ok))}")
            out[name]["judge"] = {"parsed": len(ok), "n": len(j), "acceptable": acc}
        pp = path.replace("drafts_", "probes_")
        if os.path.exists(pp):
            pr = pd.read_csv(pp, keep_default_na=False)
            print("  corruption probes (detection = marked NOT acceptable):")
            for kind in sorted(set(pr["defect"])):
                g = pr[(pr["defect"] == kind) & (pr["parsed"] == True)]  # noqa: E712
                caught = sum(1 for _, r in g.iterrows() if str(r["acceptable"]).lower() == "false")
                print(f"    {kind:<20} {ci(caught, len(g))}  n={len(g)}")
            if dry:
                print("    (dry run: the stub always answers acceptable - plumbing only)")
    if not out:
        print("\n  no draft files yet - run scripts/draft.py")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intent", action="store_true")
    ap.add_argument("--escalation", action="store_true")
    ap.add_argument("--reasons", action="store_true")
    ap.add_argument("--handoff", action="store_true")
    ap.add_argument("--drafts", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()
    if not (args.intent or args.escalation or args.reasons or args.handoff
            or args.drafts or args.all):
        ap.error("pass --intent, --escalation, --reasons, --handoff, --drafts or --all")
    gold, subset, family = load_gold()
    out: dict[str, Any] = {}
    if args.intent or args.all:
        out["intent"] = eval_intent(gold, subset, family)
    if args.escalation or args.all:
        out["escalation"] = eval_escalation(gold, subset)
    if args.reasons or args.all:
        out["reasons"] = eval_reasons(gold)
    if args.handoff or args.all:
        out["handoff"] = eval_handoff(gold)
    if args.drafts or args.all:
        out["drafts"] = eval_drafts(gold)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[json] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
