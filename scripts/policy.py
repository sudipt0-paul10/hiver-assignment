#!/usr/bin/env python3
"""Escalation policies and the mandatory trivial baselines (D2, D14, D23).

    python scripts/policy.py --all --split golden

Writes outputs/preds/escalation_<name>_<split>.csv with columns
`id,decision,score`. `decision` is ESCALATE / AUTO_OK; `score` is P(ESCALATE)
where the policy produces one and 0/1 where it does not, so the risk-coverage
curve (D5) can be drawn for the scored policies and a single point for the rest.

WHY THE LOOKUP BASELINE IS MANDATORY
------------------------------------
In the golden set, intent very nearly determines escalation: a lookup from intent
to its majority decision reproduces 93.5% of human decisions in-sample
(docs/golden_results.md 7.1). Any escalation model must therefore be reported
against a lookup, or a model that has merely learned the intent will read as a
safety result.

Two lookups exist and they are NOT interchangeable:

- `lookup-codebook` - built from the `escalation_role` field in
  docs/codebook.json (`independent` -> ESCALATE, `inherits` -> AUTO_OK), applied
  to a PREDICTED intent. Sees no human label anywhere. This is the fair baseline.
- `lookup-oracle`   - each intent mapped to its majority decision **in the golden
  set itself**, applied to the GOLD intent. Fitted on the evaluation labels twice
  over. This is a CEILING, never a baseline, and is always printed with that
  warning.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate_pilot as AP  # noqa: E402
import classify as CL  # noqa: E402
import sample_golden as SG  # noqa: E402  (FAILED_SELF_SERVICE proxy)
import corpus as C  # noqa: E402
import profile as P  # noqa: E402

PRED_DIR = CL.PRED_DIR
DM_FOLLOWUP = r"\b(?:dm(?:'?d| sent| you)|sent (?:you )?a dm|check your dm|answer my dm)\b"

# ---------------------------------------------------------------------------
# Reason codes (D9). Deterministic, ordered, first match wins.
#
# Inputs are the PREDICTED intent and observable text only. No golden label is
# read at any point, and no threshold or pattern here was chosen by looking at
# how well it scores - the ordering is the codebook's, not a fitted one.
#
# The order encodes two things already fixed elsewhere:
#   * BR-9's safety-first tie-break - when several reasons apply, the one with
#     independent escalation authority wins;
#   * D9's severity ordering - exposure and safety outrank convenience.
# `ambiguous` is the documented terminal fallback: a decision to escalate must
# carry a reason, and "we cannot tell why this is risky" is itself a reason not
# to act autonomously. How often it fires is reported as a rule-coverage
# diagnostic, not hidden.
# ---------------------------------------------------------------------------
PII = (
    r"(?:[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"          # email address
    r"|\b\d[\d \-]{11,}\d\b"                                        # long digit run (card/acct)
    r"|\b(?:my|the) (?:email|e-mail|address|phone|number) is\b)"
)
ANGER = P.OPENER_TOPICS["anger/profanity"]
OUT_OF_SCOPE = (
    r"\b(?:upload (?:my|our|his|her|their|the) (?:music|song|album|track)"
    r"|spotify for artists|distribut(?:e|ion|or)|get (?:my|our) (?:music|songs) on"
    r"|submit (?:my|our) (?:music|song|album)|as an artist)\b"
)

REASON_RULES: list[tuple[str, str]] = [
    ("public_pii", PII),
    ("safety_abuse", P.MUST_PROXY["safety"]),
    ("legal_rights", P.MUST_PROXY["legal_rights"]),
    ("security", P.MUST_PROXY["security_privacy"]),
    ("payment_dispute", P.MUST_PROXY["payment_dispute"]),
    ("account_access", P.MUST_PROXY["account_access"]),
    ("existing_case", DM_FOLLOWUP),
    ("failed_self_service", SG.FAILED_SELF_SERVICE),
    ("high_frustration", ANGER),
    ("out_of_scope", OUT_OF_SCOPE),
]
# Used only when no text rule fires: fall back to what the predicted intent implies.
INTENT_FALLBACK_REASON = {
    "account_access": "account_access",
    "billing_subscription": "payment_dispute",
    "other_unclear": "ambiguous",
}
TERMINAL_FALLBACK = "ambiguous"


def assign_reasons(text: pd.Series, decision: list[str],
                   pred_intent: list[str] | None = None) -> tuple[list[str], Counter]:
    """One reason per ESCALATE, empty string per AUTO_OK. Deterministic.

    Returns the reasons plus a counter of which rule fired, so rule coverage is
    measurable rather than assumed.
    """
    hits = {name: text.str.contains(pat, case=False, regex=True).tolist()
            for name, pat in REASON_RULES}
    fired: Counter = Counter()
    out: list[str] = []
    for i, dec in enumerate(decision):
        if dec != "ESCALATE":
            out.append("")
            continue
        chosen = ""
        for name, _ in REASON_RULES:
            if hits[name][i]:
                chosen = name
                break
        if not chosen:
            intent = pred_intent[i] if pred_intent else None
            chosen = INTENT_FALLBACK_REASON.get(intent or "", TERMINAL_FALLBACK)
            fired[f"fallback:{chosen}"] += 1
        else:
            fired[f"rule:{chosen}"] += 1
        out.append(chosen)
    return out, fired


def codebook_roles(path: str = "docs/codebook.json") -> dict[str, str]:
    cb = json.load(open(path, encoding="utf-8"))
    return {v["name"]: v["escalation_role"] for v in cb["intent"]["values"]}


def golden_frame() -> pd.DataFrame:
    with open("data/golden/golden_annotated.csv", encoding="utf-8-sig", newline="") as fh:
        return pd.DataFrame(list(csv.DictReader(fh)))


def _write(name: str, split: str, ids: list[str], decision: list[str],
           score: list[float], meta: dict[str, Any],
           text: pd.Series | None = None,
           pred_intent: list[str] | None = None) -> str:
    os.makedirs(PRED_DIR, exist_ok=True)
    out = os.path.join(PRED_DIR, f"escalation_{name}_{split}.csv")
    if text is not None:
        reasons, fired = assign_reasons(text, decision, pred_intent)
        meta["reason_stage"] = {
            "construction": "deterministic ordered rules over MUST proxies, a PII pattern, "
                            "the DM-follow-up marker, the failed-self-service proxy, an "
                            "anger pattern and an out-of-scope pattern; ties broken by BR-9 "
                            "(independent authority first). No golden label is read and "
                            "nothing was tuned.",
            "rule_activations": dict(fired),
        }
    else:
        reasons = [""] * len(ids)
    frame = pd.DataFrame({"id": ids, "decision": decision, "score": score,
                          "reason": reasons})
    frame.to_csv(out, index=False, encoding="utf-8")
    with open(out.replace(".csv", ".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    n_esc = sum(1 for d in decision if d == "ESCALATE")
    print(f"[{name}] {len(ids)} rows -> {out}   ESCALATE {n_esc}/{len(ids)} "
          f"({n_esc/len(ids):.1%})")
    print(f"  construction: {meta['construction']}")
    if meta.get("uses_human_labels"):
        print("  *** USES GOLDEN LABELS - ceiling, not a baseline ***")
    return out


def build_all(split: str, csv_path: str) -> int:
    gold = golden_frame()
    ids = list(gold["id"])
    text = gold["text"]
    roles = codebook_roles()
    # a neutral intent source for the policies that have no intent stage of their own
    rule_intent = list(CL.seed_labels(text))

    # --- trivial constants -------------------------------------------------
    _write("always-escalate", split, ids, ["ESCALATE"] * len(ids), [1.0] * len(ids),
           {"construction": "constant ESCALATE; coverage 0% by definition",
            "uses_human_labels": False}, text, rule_intent)
    _write("never-escalate", split, ids, ["AUTO_OK"] * len(ids), [0.0] * len(ids),
           {"construction": "constant AUTO_OK; coverage 100%, unsafe-auto = the base rate",
            "uses_human_labels": False}, text, rule_intent)

    # --- keyword rule: MUST proxies + DM-follow-up marker -------------------
    kw = pd.Series(False, index=text.index)
    for pat in P.MUST_PROXY.values():
        kw |= text.str.contains(pat, case=False, regex=True)
    kw |= text.str.contains(DM_FOLLOWUP, case=False, regex=True)
    _write("keyword", split, ids, ["ESCALATE" if v else "AUTO_OK" for v in kw],
           [1.0 if v else 0.0 for v in kw],
           {"construction": "union of profile.MUST_PROXY regexes plus the DM-follow-up "
                            "marker, all written against train text; nothing fitted",
            "uses_human_labels": False}, text, rule_intent)

    # --- codebook lookup over PREDICTED intents (fair, leak-free) ----------
    for src in ("rule", "distant-lr"):
        path = os.path.join(PRED_DIR, f"intent_{src}_{split}.csv")
        if not os.path.exists(path):
            print(f"  (skipping lookup-codebook:{src} - run classify.py --predict {src} first)")
            continue
        pred = pd.read_csv(path).set_index("id")["pred_intent"]
        dec = [("ESCALATE" if roles.get(pred.get(i, "other_unclear")) == "independent"
                else "AUTO_OK") for i in ids]
        _write(f"lookup-codebook-{src}", split, ids, dec,
               [1.0 if d == "ESCALATE" else 0.0 for d in dec],
               {"construction": f"codebook escalation_role applied to the '{src}' predicted "
                                f"intent; no human label enters either stage",
                "intent_source": src, "uses_human_labels": False},
               text, [pred.get(i, "other_unclear") for i in ids])

    # --- scored policy: distant-lr mass on independent-role intents --------
    lr_path = os.path.join(PRED_DIR, f"intent_distant-lr_{split}.csv")
    if os.path.exists(lr_path):
        pairs = C.opener_pairs(C.load(csv_path))
        train = pairs[(pairs["split"] == "train") & (pairs["clean"].str.len() > 0)]
        model = CL.DistantLR().fit(train)
        _, proba, classes = model.predict(gold["text"])
        idx = [i for i, c in enumerate(classes) if roles.get(c) == "independent"]
        score = proba[:, idx].sum(1)
        _write("distant-lr-score", split, ids,
               ["ESCALATE" if s >= 0.5 else "AUTO_OK" for s in score],
               [round(float(s), 6) for s in score],
               {"construction": "distant-supervision LR probability mass on the intents the "
                                "codebook marks escalation_role=independent; threshold 0.5 "
                                "fixed a priori, NOT tuned (no labelled dev set exists, D23)",
                "proxy_baseline": True, "uses_human_labels": False,
                "note": "the only policy here with a continuous score, so the only one that "
                        "can draw a risk-coverage curve rather than a single point"},
               text, list(CL.DistantLR().fit(train).predict(gold["text"])[0]))

    # --- oracle ceiling: majority decision per GOLD intent -----------------
    by: dict[str, Counter] = defaultdict(Counter)
    for _, r in gold.iterrows():
        by[r["intent"]][r["escalation"]] += 1
    maj = {k: v.most_common(1)[0][0] for k, v in by.items()}
    dec = [maj[r["intent"]] for _, r in gold.iterrows()]
    _write("lookup-oracle", split, ids, dec, [1.0 if d == "ESCALATE" else 0.0 for d in dec],
           {"construction": "each intent mapped to its MAJORITY escalation decision in the "
                            "golden set, applied to the GOLD intent",
            "uses_human_labels": True,
            "warning": "fitted on the evaluation labels; a CEILING, never a baseline. Reports "
                       "the score an escalation model would reach if it learned the intent "
                       "perfectly and nothing else.",
            "mapping": maj}, text, list(gold["intent"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", default="golden")
    ap.add_argument("--csv", default="twcs/twcs.csv")
    args = ap.parse_args()
    if not args.all:
        ap.error("pass --all")
    return build_all(args.split, args.csv)


if __name__ == "__main__":
    raise SystemExit(main())
