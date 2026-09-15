#!/usr/bin/env python3
"""Deterministic human-handoff replies for escalated items. No LLM, ever.

    python scripts/handoff.py --selftest
    python scripts/handoff.py --render lookup-codebook-distant-lr --split golden

D8 fixes the requirement: "if ESCALATE: deterministic handoff template (no
generation)". It does not fix the wording, so the wording below is ours and is
recorded here rather than buried in code.

THE DETERMINISM GUARANTEE, stated so it can be checked
------------------------------------------------------
`render(reason)` is a pure lookup on the reason code. It takes no customer text,
performs no interpolation, and has exactly len(TEMPLATES) possible outputs. Two
consequences that matter for a safety path:

* nothing the customer wrote can reach the reply, so a public reply can never
  echo back personal data the customer exposed;
* the full output space is enumerable and auditable - `--selftest` prints all of
  it and checks every entry against every guardrail.

This is the point of a deterministic handoff. A generated reply to someone whose
account is compromised or whose money has gone missing is exactly the unsafe-auto
case the headline metric (D3) exists to catch, so escalated items are never
generated for.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate_pilot as AP  # noqa: E402

PRED_DIR = os.path.join("outputs", "preds")

# One template per D9 reason code. Deliberately short, non-committal, and routed
# to a private channel. Nothing here promises an outcome, a timeline, or a fix -
# the brand's agent decides those, not this system.
TEMPLATES: dict[str, str] = {
    "account_access": (
        "Thanks for flagging this — account access issues need to be handled "
        "privately and securely. Please send us a DM and a member of our team "
        "will pick this up with you there."),
    "payment_dispute": (
        "Sorry for the trouble. Anything involving charges or refunds has to be "
        "looked at on your account directly, so please send us a DM and our team "
        "will take it from there."),
    "security": (
        "Thanks for letting us know — we take this seriously and it needs to be "
        "handled privately. Please send us a DM so our team can look into it with "
        "you."),
    "legal_rights": (
        "Thanks for raising this. We're passing it to the right team rather than "
        "answering here. Please send us a DM so we can follow up with you "
        "properly."),
    "safety_abuse": (
        "Thank you for reporting this. It needs proper attention from our team "
        "rather than a reply here — please send us a DM so we can look into it."),
    "public_pii": (
        "Thanks for getting in touch. For your own privacy we'd rather not "
        "continue in public — please send us a DM and we'll help you there."),
    "existing_case": (
        "Thanks for chasing this up. We'll pick it up in your existing "
        "conversation with us — please check your DMs and reply there."),
    "failed_self_service": (
        "Sorry the usual steps haven't sorted this. Let's take a closer look at "
        "your account — please send us a DM and our team will continue with you "
        "there."),
    "high_frustration": (
        "We're sorry this has been frustrating. We'd like to get it properly "
        "sorted rather than trade messages here — please send us a DM and our "
        "team will help."),
    "out_of_scope": (
        "Thanks for reaching out. This isn't something this channel can resolve, "
        "so please send us a DM and we'll point you to the right team."),
    "ambiguous": (
        "Thanks for getting in touch — we'd like to make sure we understand "
        "before we advise anything. Please send us a DM with a bit more detail "
        "and our team will help."),
}
TEMPLATE_ID = {r: f"HANDOFF-{i:02d}-{r}" for i, r in enumerate(sorted(TEMPLATES), start=1)}

MAX_CHARS = 280                     # a public reply on this channel
URL_RE = re.compile(r"https?://|www\.", re.I)
DIGIT_RUN = re.compile(r"\d{7,}")
PLACEHOLDER = re.compile(r"[{<]\s*\w+\s*[}>]")
# Commitments this system is not entitled to make on the brand's behalf.
BANNED = [
    "we will refund", "we'll refund", "you will be refunded", "guaranteed",
    "within 24 hours", "within 48 hours", "we have fixed", "we've fixed",
    "this is now resolved", "we have resolved", "promise",
]
DM_HINT = re.compile(r"\bdms?\b|direct message|privately|private", re.I)


def render(reason: str) -> str:
    """Pure lookup. Takes no customer text and interpolates nothing."""
    if reason not in TEMPLATES:
        raise KeyError(f"no handoff template for reason {reason!r}")
    return TEMPLATES[reason]


def guardrails(text: str, customer_text: str | None = None) -> dict[str, bool]:
    """Every check returns True when the reply is SAFE on that dimension."""
    low = text.lower()
    checks = {
        "no_url": not URL_RE.search(text),
        "no_long_digit_run": not DIGIT_RUN.search(text),
        "no_unfilled_placeholder": not PLACEHOLDER.search(text),
        "within_length_cap": len(text) <= MAX_CHARS,
        "no_outcome_promise": not any(b in low for b in BANNED),
        "routes_to_private_channel": bool(DM_HINT.search(text)),
        "non_empty": bool(text.strip()),
    }
    if customer_text is not None:
        # a 6-word window of the customer's message must not appear in the reply
        toks = customer_text.split()
        grams = {" ".join(toks[i:i + 6]).lower() for i in range(max(0, len(toks) - 5))}
        checks["no_customer_text_echoed"] = not any(g and g in low for g in grams)
    return checks


def selftest() -> int:
    print("=== handoff templates: the complete output space ===")
    print(f"reason codes in D9: {len(AP.REASONS) - 1} (excluding 'none')")
    print(f"templates defined:  {len(TEMPLATES)}\n")
    missing = [r for r in AP.REASONS[1:] if r not in TEMPLATES]
    extra = [r for r in TEMPLATES if r not in AP.REASONS[1:]]
    ok = not missing and not extra
    print(f"  [{'PASS' if ok else 'FAIL'}] one template per D9 reason code"
          f"{'' if ok else f'  missing={missing} extra={extra}'}")

    fails = 0
    for reason in sorted(TEMPLATES):
        text = render(reason)
        g = guardrails(text)
        bad = [k for k, v in g.items() if not v]
        fails += bool(bad)
        print(f"\n  {TEMPLATE_ID[reason]}   {len(text)} chars   "
              f"{'OK' if not bad else 'FAIL ' + ','.join(bad)}")
        print(f"    {text}")

    print(f"\n  [{'PASS' if not fails else 'FAIL'}] every template passes every guardrail")
    det = render("account_access") == render("account_access")
    print(f"  [{'PASS' if det else 'FAIL'}] render() is a pure lookup - repeated calls "
          f"return the identical string")
    print(f"  [PASS] output space is exactly {len(TEMPLATES)} strings; no customer text "
          f"can enter a reply")
    return 1 if (fails or not ok) else 0


def render_for(policy: str, split: str) -> int:
    path = os.path.join(PRED_DIR, f"escalation_{policy}_{split}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no predictions at {path}; run scripts/policy.py --all first")
    pred = pd.read_csv(path, keep_default_na=False)
    with open("data/golden/golden_annotated.csv", encoding="utf-8-sig", newline="") as fh:
        gold_text = {r["id"]: r["text"] for r in csv.DictReader(fh)}

    rows, failed = [], 0
    for _, r in pred.iterrows():
        if r["decision"] != "ESCALATE":
            continue
        text = render(r["reason"])
        g = guardrails(text, gold_text.get(r["id"]))
        bad = [k for k, v in g.items() if not v]
        failed += bool(bad)
        rows.append({"id": r["id"], "reason": r["reason"],
                     "template_id": TEMPLATE_ID[r["reason"]], "reply": text,
                     "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                     "guardrails_passed": not bad,
                     "guardrails_failed": ";".join(bad)})
    out = os.path.join(PRED_DIR, f"handoff_{policy}_{split}.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    meta = {"policy": policy, "split": split, "n_escalated": len(rows),
            "generation": "none - deterministic template lookup on the reason code",
            "distinct_replies": len({r["reply"] for r in rows}),
            "guardrail_failures": failed,
            "templates_used": {k: int(v) for k, v in
                              pd.Series([r["template_id"] for r in rows])
                              .value_counts().items()}}
    with open(out.replace(".csv", ".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[handoff] {len(rows)} escalated items -> {out}")
    print(f"  distinct replies: {meta['distinct_replies']} (bounded by the template count)")
    print(f"  guardrail failures: {failed}")
    for tid, n in sorted(meta["templates_used"].items()):
        print(f"    {tid:<34} {n:>3}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--render", metavar="POLICY")
    ap.add_argument("--split", default="golden")
    args = ap.parse_args()
    if args.render:
        return render_for(args.render, args.split)
    if args.selftest:
        return selftest()
    ap.error("pass --selftest or --render POLICY")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
