#!/usr/bin/env python3
"""D8 drafting: grounded reply for AUTO_OK items, deterministic handoff for ESCALATE.

    python scripts/draft.py --dry-run --policy lookup-codebook-distant-lr   # no API
    python scripts/draft.py --show-prompt
    python scripts/draft.py --policy lookup-codebook-distant-lr --allow-api  # spends

The escalation path never reaches the model. Escalated items get the fixed
template from scripts/handoff.py; only auto-handled items are drafted, and every
draft is grounded in retrieved TRAIN replies (D8) with the leakage guards from
scripts/retrieval.py applied per item.

GUARDRAILS, and what happens when one trips
-------------------------------------------
A generated reply is checked with the same guardrail set as the handoff templates
plus two that only matter for generation: it must not invent a URL, and it must
not invent a number that looks like an amount, order or account reference - the
model has no access to the customer's account and anything specific it produces
is a hallucination. **A draft that fails any guardrail is discarded and the item
falls back to the deterministic handoff template.** Failing safe into a handoff
is the only behaviour consistent with D3: an unsafe auto-reply is the exact error
the headline metric exists to catch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify as CL  # noqa: E402
import corpus as C  # noqa: E402
import handoff as H  # noqa: E402
import providers as PR  # noqa: E402
import retrieval as R  # noqa: E402

PRED_DIR = CL.PRED_DIR
K_EVIDENCE = 3
DRAFT_PARAMS: dict[str, Any] = {"max_tokens": 200, "temperature": 0.0}
MAX_CHARS = H.MAX_CHARS
INVENTED_NUMBER = re.compile(r"[$£€]\s?\d|\b\d{4,}\b|\border\s*#?\d|\bref(?:erence)?\s*#?\d", re.I)

# TWCS anonymises customer handles as pure digits - @115888, @116130, @448281 - so a
# reply that addresses the customer by handle, which is ordinary Twitter behaviour and
# is what the retrieved evidence demonstrates, was tripping the \b\d{4,}\b branch of
# INVENTED_NUMBER above. A digit run written immediately after "@" is a mention, not a
# reference number, so mentions are removed before the check runs. The pattern is
# deliberately narrow: only "@" followed by digits to the end of the token. Every other
# number - a bare 8842190, an amount, an order or reference id - still trips the guard.
ANON_MENTION = re.compile(r"@\d+\b")

PROMPT = """You are drafting a PUBLIC reply from Spotify's support account on Twitter.

Below are real past customer messages and the replies Spotify actually sent. Use
them as evidence for how this brand handles this kind of issue. Follow their
style and the kind of help they give.

{evidence}

RULES - all of them are hard:
- 280 characters maximum.
- Do not invent URLs, links, amounts, dates, order numbers or account details.
  You cannot see this customer's account.
- Do not promise a refund, a fix, or a timeline.
- Do not repeat the customer's message back to them.
- If the issue genuinely cannot be resolved with a public reply, say so briefly
  and ask them to send a DM.
- Reply with the message text only, nothing else.

CUSTOMER MESSAGE:
{message}"""




def build_prompt(evidence: list[dict[str, Any]]) -> str:
    """Evidence is interpolated at build time; {message} stays for the cache key."""
    lines = []
    for i, h in enumerate(evidence, 1):
        lines.append(f"[past case {i}]\ncustomer: {h['opener']}\nspotify: {h['reply']}")
    return PROMPT.replace("{evidence}", "\n\n".join(lines) if lines
                          else "(no similar past case was found)")


def guardrails(text: str, customer_text: str) -> dict[str, bool]:
    """Guardrails for a GENERATED public reply.

    Two differences from the handoff set, both about which checks apply here
    rather than about what any check does:

    * private-channel routing is not required - an AUTO_OK draft is a public
      reply by construction (see handoff.guardrails);
    * anonymised @mentions are excluded from the invented-number check.
    """
    g = H.guardrails(text, customer_text, require_private_routing=False)
    g["no_invented_number"] = not INVENTED_NUMBER.search(ANON_MENTION.sub(" ", text))
    return g


def run(policy: str, split: str, csv_path: str, allow_api: bool,
        provider_name: str, base_url: str | None, limit: int | None,
        model: str | None = None) -> int:
    provider = PR.get_provider(
        provider_name, CL.model_for("drafting", provider_name, model), base_url)
    is_mock = bool(getattr(provider, "is_mock", False))
    esc_path = os.path.join(PRED_DIR, f"escalation_{policy}_{split}.csv")
    if not os.path.exists(esc_path):
        raise SystemExit(f"no policy predictions at {esc_path}; run scripts/policy.py --all")
    decisions = pd.read_csv(esc_path, keep_default_na=False)
    with open("data/golden/golden_annotated.csv", encoding="utf-8-sig", newline="") as fh:
        items = [{"id": r["id"], "text": r["text"]} for r in csv.DictReader(fh)]
    if limit:
        items = items[:limit]
    guards = R.golden_guards()

    pairs = C.opener_pairs(C.load(csv_path))
    retr = R.Retriever(C.retrieval_corpus(pairs))

    dec = dict(zip(decisions["id"], decisions["decision"]))
    rsn = dict(zip(decisions["id"], decisions["reason"]))
    rows, fallbacks = [], 0
    for it in items:
        gid = it["id"]
        if dec.get(gid) == "ESCALATE":
            rows.append({"id": gid, "route": "handoff", "source": "template",
                         "template_id": H.TEMPLATE_ID[rsn[gid]], "reply": H.render(rsn[gid]),
                         "n_evidence": 0, "guardrails_passed": True, "guardrails_failed": "",
                         "cached": ""})
            continue
        g = guards[gid]
        hits, _ = retr.search(it["text"], k=K_EVIDENCE,
                              exclude_customers=[g["customer_id"]],
                              exclude_threads=[g["thread_id"]], exclude_tweets=[g["tweet_id"]])
        prompt = build_prompt(hits)
        blob = CL.cached_call("draft", provider, prompt, it["text"], DRAFT_PARAMS, allow_api)
        text, cached = blob["text"], blob["cached"]
        gr = guardrails(text, it["text"])
        bad = [k for k, v in gr.items() if not v]
        if bad:                                  # fail safe: a bad draft becomes a handoff
            fallbacks += 1
            reason = rsn.get(gid) or "ambiguous"
            rows.append({"id": gid, "route": "handoff_fallback", "source": "template",
                         "template_id": H.TEMPLATE_ID[reason], "reply": H.render(reason),
                         "n_evidence": len(hits), "guardrails_passed": False,
                         "guardrails_failed": ";".join(bad), "cached": cached})
        else:
            rows.append({"id": gid, "route": "auto_reply", "source": "llm_draft",
                         "template_id": "", "reply": text, "n_evidence": len(hits),
                         "guardrails_passed": True, "guardrails_failed": "", "cached": cached})

    suffix = "_mock" if is_mock else ""
    out = os.path.join(PRED_DIR, f"drafts_{policy}_{split}{suffix}.csv")
    os.makedirs(PRED_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    meta = {"policy": policy, "split": split, "is_mock": is_mock,
            "provider": provider.name, "model": provider.model, "params": DRAFT_PARAMS,
            "k_evidence": K_EVIDENCE, "n_items": len(rows),
            "n_auto_reply": sum(1 for r in rows if r["route"] == "auto_reply"),
            "n_handoff": sum(1 for r in rows if r["route"] == "handoff"),
            "n_guardrail_fallback": fallbacks,
            "usage": CL.usage_summary()}
    with open(out.replace(".csv", ".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[draft] {len(rows)} items -> {out}")
    print(f"  auto_reply {meta['n_auto_reply']}   handoff {meta['n_handoff']}   "
          f"guardrail fallback {fallbacks}")
    print(f"  usage: {json.dumps(meta['usage'])}")
    if is_mock:
        print("  *** MOCK PROVIDER - stub replies, NOT a result. Pipeline exercise only. ***")
    return 0


def selftest() -> int:
    """Guardrails must reject the failure modes generation actually produces."""
    ok = True
    cust = "my premium account got charged twice this month and I want it sorted"

    def chk(good: bool, label: str) -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}")

    print("=== draft guardrail selftest (no API) ===")
    good = ("Sorry about that! Try signing out and back in, and check the app is up to "
            "date. If it keeps happening, send us a DM and we'll take a look.")
    chk(all(guardrails(good, cust).values()), "a clean grounded draft passes every guardrail")

    cases = [
        ("invented URL", "Please see https://spotify.com/fix for the steps.", "no_url"),
        ("invented amount", "We've refunded £47.30 to your card.", "no_invented_number"),
        ("invented order ref", "Your reference 8842190 is being processed.", "no_invented_number"),
        ("outcome promise", "We will refund you within 24 hours, promise.", "no_outcome_promise"),
        ("too long", "x" * (MAX_CHARS + 1), "within_length_cap"),
        ("empty", "   ", "non_empty"),
        ("echoes the customer", "You said: my premium account got charged twice this "
                                "month and I want it sorted. DM us.", "no_customer_text_echoed"),
        ("unfilled placeholder", "Hi {name}, please DM us.", "no_unfilled_placeholder"),
    ]
    for label, text, expect in cases:
        g = guardrails(text, cust)
        chk(g.get(expect) is False, f"{label} is caught by {expect}")

    # --- regressions for the two guardrail-APPLICATION fixes -------------------
    print("\n  -- anonymised @handles vs genuinely invented numbers --")
    chk(guardrails("@115888 Hey! Can you tell us your device and OS?", cust)
        ["no_invented_number"],
        "anonymised handle @115888 does NOT trip no_invented_number")
    chk(guardrails("@116130 @448281 thanks for flagging, we're on it.", cust)
        ["no_invented_number"],
        "several anonymised handles do NOT trip no_invented_number")
    chk(not guardrails("Your reference 8842190 is being processed.", cust)
        ["no_invented_number"],
        "a genuinely invented reference 8842190 STILL trips no_invented_number")
    chk(not guardrails("We've refunded £47.30 to your card.", cust)["no_invented_number"],
        "an invented amount STILL trips no_invented_number")
    chk(not guardrails("@115888 your order 9931002 shipped.", cust)["no_invented_number"],
        "a handle plus an invented number still trips it (handle is not a shield)")

    print("\n  -- public drafts vs handoff replies on private routing --")
    public = ("Try the Repeat option to switch that off, and Shuffle to mix up the "
              "playlist order.")
    chk(all(guardrails(public, cust).values()),
        "a legitimate public AUTO_OK reply with no DM request now passes")
    chk("routes_to_private_channel" not in guardrails(public, cust),
        "routes_to_private_channel is not applied on the drafting path")
    chk(not H.guardrails("Thanks, we have noted that.")["routes_to_private_channel"],
        "a handoff reply with no private routing STILL fails (guard intact by default)")
    chk(all(H.guardrails(H.render("account_access")).values()),
        "every handoff template still routes privately and passes its own set")

    print(f"\n{'SELFTEST PASSED' if ok else 'SELFTEST FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="lookup-codebook-distant-lr")
    ap.add_argument("--split", default="golden")
    ap.add_argument("--csv", default="twcs/twcs.csv")
    ap.add_argument("--provider", default=PR.DEFAULT_PROVIDER, choices=list(PR.PROVIDERS))
    ap.add_argument("--base-url")
    ap.add_argument("--model", help="model id; overrides the provider default")
    ap.add_argument("--allow-api", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.show_prompt:
        print(build_prompt([{"opener": "<past customer message>", "reply": "<spotify reply>"}]))
        return 0
    return run(args.policy, args.split, args.csv, args.allow_api,
               args.provider, args.base_url, args.limit, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
