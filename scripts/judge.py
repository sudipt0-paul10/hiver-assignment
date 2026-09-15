#!/usr/bin/env python3
"""D11 LLM-as-judge for reply quality, plus the corruption probes that qualify it.

    python scripts/judge.py --selftest                      # parser + probe construction, no API
    python scripts/judge.py --dry-run --policy <p>           # pipeline mechanics, no API
    python scripts/judge.py --show-prompt
    python scripts/judge.py --policy <p> --allow-api         # spends
    python scripts/judge.py --probes --policy <p> --allow-api
    python scripts/judge.py --rating-sheet 60 --policy <p>   # blank sheet for the human pass
    python scripts/judge.py --agreement --policy <p>         # once the sheet is filled

Rubric is D11's, unchanged: addresses need, grounded in historical evidence,
appropriate routing, tone, and an overall acceptability verdict.

TWO TESTS, AND THE SECOND IS THE SHARPER ONE
--------------------------------------------
Correlation with human ratings says the judge agrees with us. The corruption
probes say whether it can detect defects at all: replies are deliberately broken
in four known ways - wrong intent, fabricated facts, missing escalation, hostile
tone - and a judge that marks those acceptable is disqualified regardless of how
well it correlates. The corruptions are built deterministically in this file; no
model is involved in creating them.

The judge never sees the human labels, the intent, or the escalation decision -
only the customer message, the evidence, and the reply.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify as CL  # noqa: E402
import profile as P  # noqa: E402

PRED_DIR = CL.PRED_DIR
JUDGE_PARAMS: dict[str, Any] = {"max_tokens": 300, "temperature": 0.0}
DIMENSIONS = ["addresses_need", "grounded", "routing", "tone"]

PROMPT = """You are rating the quality of a support reply. Score strictly.

Rate each dimension 1-5 (1 = very poor, 5 = very good):
- addresses_need: does the reply engage with what the customer actually asked?
- grounded: is it consistent with how this brand handled similar cases, and free
  of invented facts, amounts, dates or links?
- routing: is the choice to answer publicly versus hand off to a private channel
  the right one for this issue?
- tone: is it appropriate, respectful and professional?

Then give an overall verdict: would it be acceptable to send this reply to this
customer as-is? A reply with a fabricated fact, a missed escalation of an account
or payment issue, or a hostile tone is NOT acceptable however good it otherwise
reads.

Answer with JSON only, no other text:
{"addresses_need": n, "grounded": n, "routing": n, "tone": n,
 "acceptable": true|false, "why": "one short sentence"}

{message}"""


def build_judge_input(customer: str, reply: str) -> str:
    return f"CUSTOMER MESSAGE:\n{customer}\n\nPROPOSED REPLY:\n{reply}"


def parse(text: str) -> dict[str, Any] | None:
    """Parse the judge's JSON. Returns None when it cannot be trusted."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        d = json.loads(t[a:b + 1])
    except json.JSONDecodeError:
        return None
    if not all(k in d for k in DIMENSIONS + ["acceptable"]):
        return None
    for k in DIMENSIONS:
        if not isinstance(d[k], int) or not 1 <= d[k] <= 5:
            return None
    if not isinstance(d["acceptable"], bool):
        return None
    return d


# --- corruption probes: built here, deterministically, with no model ---------
def corrupt(kind: str, reply: str, other_reply: str) -> str:
    if kind == "wrong_intent":
        return other_reply
    if kind == "fabricated_fact":
        return (reply + " Your refund of £47.30 was processed on 12 March and your "
                        "account has been upgraded.")
    if kind == "missing_escalation":
        return "Thanks! That should sort itself out shortly, no need to worry."
    if kind == "hostile_tone":
        return "Honestly, this is user error. " + reply
    raise KeyError(kind)


PROBE_KINDS = ["wrong_intent", "fabricated_fact", "missing_escalation", "hostile_tone"]
STUB_JUDGE = '{"addresses_need": 4, "grounded": 4, "routing": 4, "tone": 5, ' \
             '"acceptable": true, "why": "offline stub"}'


def _load(policy: str, split: str, dry_run: bool) -> tuple[list[dict[str, str]], pd.DataFrame]:
    suffix = "_dryrun" if dry_run else ""
    path = os.path.join(PRED_DIR, f"drafts_{policy}_{split}{suffix}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no drafts at {path}; run scripts/draft.py first")
    with open("data/golden/golden_annotated.csv", encoding="utf-8-sig", newline="") as fh:
        items = [{"id": r["id"], "text": r["text"]} for r in csv.DictReader(fh)]
    return items, pd.read_csv(path, keep_default_na=False)


def _judge_one(customer: str, reply: str, allow_api: bool, dry_run: bool) -> tuple[Any, str]:
    if dry_run:
        return parse(STUB_JUDGE), "dry-run"
    blob = CL.cached_call("judge", CL.MODELS["judge"], PROMPT,
                          build_judge_input(customer, reply), JUDGE_PARAMS, allow_api)
    return parse(blob["text"]), blob["cached"]


def run_judge(policy: str, split: str, allow_api: bool, dry_run: bool, limit: int | None) -> int:
    items, drafts = _load(policy, split, dry_run)
    text = {i["id"]: i["text"] for i in items}
    rows = []
    for _, d in (drafts.head(limit) if limit else drafts).iterrows():
        v, cached = _judge_one(text[d["id"]], d["reply"], allow_api, dry_run)
        rows.append({"id": d["id"], "route": d["route"], "parsed": v is not None,
                     **({k: v[k] for k in DIMENSIONS} if v else {k: "" for k in DIMENSIONS}),
                     "acceptable": v["acceptable"] if v else "",
                     "why": v["why"] if v else "", "cached": cached})
    suffix = "_dryrun" if dry_run else ""
    out = os.path.join(PRED_DIR, f"judge_{policy}_{split}{suffix}.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    bad = sum(1 for r in rows if not r["parsed"])
    meta = {"policy": policy, "dry_run": dry_run,
            "model": None if dry_run else CL.MODELS["judge"], "params": JUDGE_PARAMS,
            "n": len(rows), "unparseable": bad,
            "usage": None if dry_run else CL.usage_summary()}
    with open(out.replace(".csv", ".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[judge] {len(rows)} replies -> {out}   unparseable {bad}")
    ok = [r for r in rows if r["parsed"]]
    if ok:
        acc = sum(1 for r in ok if r["acceptable"] is True)
        print(f"  acceptable {acc}/{len(ok)}")
        for dim in DIMENSIONS:
            vals = [r[dim] for r in ok]
            print(f"  mean {dim:<16} {sum(vals)/len(vals):.2f}")
    if not dry_run:
        print(f"  usage: {json.dumps(meta['usage'])}")
    return 0


def run_probes(policy: str, split: str, allow_api: bool, dry_run: bool, n: int) -> int:
    items, drafts = _load(policy, split, dry_run)
    text = {i["id"]: i["text"] for i in items}
    clean = drafts[drafts["route"] == "auto_reply"].reset_index(drop=True)
    if len(clean) < 2:
        raise SystemExit("need at least 2 auto_reply drafts to build probes")
    take = clean.head(n)
    rows = []
    for i, d in take.iterrows():
        other = clean.iloc[(int(i) + 1) % len(clean)]["reply"]
        for kind in PROBE_KINDS:
            bad = corrupt(kind, d["reply"], other)
            v, cached = _judge_one(text[d["id"]], bad, allow_api, dry_run)
            rows.append({"id": d["id"], "defect": kind, "parsed": v is not None,
                         "acceptable": v["acceptable"] if v else "",
                         "grounded": v["grounded"] if v else "",
                         "tone": v["tone"] if v else "", "cached": cached})
    suffix = "_dryrun" if dry_run else ""
    out = os.path.join(PRED_DIR, f"probes_{policy}_{split}{suffix}.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"[probes] {len(rows)} corrupted replies -> {out}")
    print("  detection = judge marks the corrupted reply NOT acceptable")
    for kind in PROBE_KINDS:
        g = [r for r in rows if r["defect"] == kind and r["parsed"]]
        if g:
            caught = sum(1 for r in g if r["acceptable"] is False)
            lo, hi = P.wilson(caught, len(g))
            print(f"  {kind:<20} {caught}/{len(g)}  [{lo:.0%}, {hi:.0%}]")
    if dry_run:
        print("  NOTE: the offline stub always answers 'acceptable', so detection cannot be")
        print("  validated without a real call. This run proves construction and plumbing only.")
    return 0


def rating_sheet(policy: str, split: str, n: int) -> int:
    """Blank sheet for D11's human reply-quality ratings. No labels, no judge output."""
    items, drafts = _load(policy, split, False)
    text = {i["id"]: i["text"] for i in items}
    import numpy as np
    rng = np.random.default_rng(11)                      # D11
    idx = rng.choice(len(drafts), size=min(n, len(drafts)), replace=False)
    rows = [{"id": drafts.iloc[int(i)]["id"], "customer_message": text[drafts.iloc[int(i)]["id"]],
             "reply": drafts.iloc[int(i)]["reply"],
             **{d: "" for d in DIMENSIONS}, "acceptable": "", "notes": ""}
            for i in sorted(idx)]
    out = os.path.join("data", "golden", f"human_ratings_{policy}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"[rating-sheet] {len(rows)} blank ratings -> {out}")
    print("  Rate each dimension 1-5 and acceptable y/n, using the same rubric as the judge.")
    print("  Do NOT look at the judge output while rating.")
    return 0


def agreement(policy: str, split: str) -> int:
    hp = os.path.join("data", "golden", f"human_ratings_{policy}.csv")
    jp = os.path.join(PRED_DIR, f"judge_{policy}_{split}.csv")
    for p_ in (hp, jp):
        if not os.path.exists(p_):
            raise SystemExit(f"missing {p_}")
    hum = pd.read_csv(hp, keep_default_na=False)
    jud = pd.read_csv(jp, keep_default_na=False).set_index("id")
    done = hum[hum["acceptable"].astype(str).str.strip() != ""]
    if not len(done):
        raise SystemExit("no human ratings filled in yet")
    pairs = [(r["id"], str(r["acceptable"]).strip().lower()[:1],
              str(jud.loc[r["id"], "acceptable"]).strip().lower()[:1])
             for _, r in done.iterrows() if r["id"] in jud.index]
    k = sum(1 for _, h, j in pairs if (h == "y") == (j == "t"))
    lo, hi = P.wilson(k, len(pairs))
    print(f"[agreement] judge vs human on 'acceptable': {k}/{len(pairs)} = "
          f"{k/len(pairs):.1%}  [{lo:.1%}, {hi:.1%}]")
    print("  Read against the D12 self-agreement ceiling (docs/retest.md), not against 100%.")
    return 0


def selftest() -> int:
    ok = True

    def chk(good: bool, label: str) -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}")

    print("=== judge selftest (no API) ===")
    print("\nparser")
    chk(parse('{"addresses_need":4,"grounded":3,"routing":5,"tone":4,'
              '"acceptable":true,"why":"ok"}') is not None, "valid JSON accepted")
    chk(parse('```json\n{"addresses_need":4,"grounded":3,"routing":5,"tone":4,'
              '"acceptable":false,"why":"x"}\n```') is not None, "fenced JSON accepted")
    chk(parse("not json at all") is None, "non-JSON rejected")
    chk(parse('{"addresses_need":9,"grounded":3,"routing":5,"tone":4,'
              '"acceptable":true,"why":"x"}') is None, "out-of-range score rejected")
    chk(parse('{"addresses_need":4,"grounded":3,"routing":5,"tone":4,"why":"x"}') is None,
        "missing verdict rejected")
    chk(parse('{"addresses_need":4,"grounded":3,"routing":5,"tone":4,'
              '"acceptable":"yes","why":"x"}') is None, "non-boolean verdict rejected")

    print("\nprobe construction")
    base, other = "Try reinstalling the app and let us know.", "Send us a DM about your playlist."
    for kind in PROBE_KINDS:
        c = corrupt(kind, base, other)
        chk(isinstance(c, str) and c != base, f"{kind} produces a changed reply")
    chk(corrupt("wrong_intent", base, other) == other, "wrong_intent swaps in another reply")
    chk("£47.30" in corrupt("fabricated_fact", base, other), "fabricated_fact injects a fake fact")
    chk(corrupt("hostile_tone", base, other).startswith("Honestly"), "hostile_tone prepends")
    print(f"\n{'SELFTEST PASSED' if ok else 'SELFTEST FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="lookup-codebook-distant-lr")
    ap.add_argument("--split", default="golden")
    ap.add_argument("--allow-api", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--rating-sheet", type=int, metavar="N")
    ap.add_argument("--agreement", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--probe-items", type=int, default=20)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.show_prompt:
        print(PROMPT.replace("{message}", build_judge_input("<customer message>", "<reply>")))
        return 0
    if args.rating_sheet:
        return rating_sheet(args.policy, args.split, args.rating_sheet)
    if args.agreement:
        return agreement(args.policy, args.split)
    if args.probes:
        return run_probes(args.policy, args.split, args.allow_api, args.dry_run, args.probe_items)
    return run_judge(args.policy, args.split, args.allow_api, args.dry_run, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
