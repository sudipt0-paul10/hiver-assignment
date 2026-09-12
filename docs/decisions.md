# Prior Architecture & Evaluation Decisions

**STATUS: PROPOSED — VALIDATE AGAINST THE ACTUAL SPEC AND DATA**

## Brand
Provisional primary: **SpotifyCares**. AmericanAir was the runner-up. Re-profile the official dataset before locking this.

## Provisional pipeline
Customer message → intent classification → escalation/action decision → historical-response retrieval → LLM draft → deterministic guardrails.

Escalated cases should receive a deterministic handoff-style response rather than pretending to resolve private/account-specific issues.

Start retrieval with TF-IDF/BM25. Add embeddings/vector search only if evaluation proves a meaningful gain.

## Intent classifier
Start classical:
- keyword rules as a simple baseline;
- TF-IDF + Logistic Regression as the main simple classifier;
- use human-labelled development data plus defensible silver signals.
Avoid thousands of LLM pseudo-labels unless evidence shows they materially help. A zero-shot LLM classifier is optional, not core.

## Provisional taxonomy
- `account_access`
- `billing_subscription`
- `playback_technical`
- `library_downloads`
- `content_availability`
- `feature_feedback`
- `dm_followup`
- `social_nonrequest`
- `referral_other`
- `unclear`

Target roughly 7–10 useful classes. Keep account access, billing/subscription, and DM/case follow-up distinct if their operational outcomes differ. Treat emotion as an attribute.

Previously proposed taxonomy rules:
- roughly 4% prevalence / ~8 examples in a 200-item random core as a minimum;
- merge classes if pilot mutual confusion is >25%, unless their escalation/action outcomes differ;
- split only when the distinction is operationally meaningful, especially if escalation differs.

## Human pilot
Run ~40 blind human examples before the final golden set. Label intent, escalation level, reason, attributes, and genuine uncertainty. Do not inspect historical brand replies while labeling. Use hesitation/confusion to improve the codebook, then lock it.

## Escalation policy
TWCS has no true escalation ground truth. Escalation labels are a defined human policy; historical DM behavior is secondary evidence only.

Provisional levels:
- `MUST`: should not be autonomously handled;
- `SHOULD`: likely needs human review;
- `AUTO_OK`: safe for autonomous public handling.

MUST candidates: account access, payment disputes, security, legal/rights, safety, existing private case.
SHOULD candidates: persistent failure, high frustration/churn with unresolved problem, out-of-scope, ambiguity.
AUTO_OK candidates: public how-to, simple clarification, known issue/content explanation, ordinary feedback, thanks/social.

Candidate reasons: `account_access`, `payment_dispute`, `security`, `legal_rights`, `safety_abuse`, `public_pii`, `existing_case`, `failed_self_service`, `high_frustration`, `out_of_scope`, `ambiguous`.

Do not force an arbitrary MUST-recall target; choose the operating point from development data and an explicit risk policy.

## Golden set
Previous proposal: 240 items:
- 200 random/core opening turns from the test window;
- 40 separate stress items.
Core: one per thread, customer-disjoint, near-duplicates removed.
Stress: 25 follow-ups + 15 rare/high-risk examples such as security/legal/non-English.
Report stress separately because it is intentionally biased.

## Split
Previously proposed Spotify opener split:
- train: Oct 1–Nov 5;
- dev: Nov 6–Nov 19;
- test: Nov 20–Dec 3.
Previous approximate opener counts: 14,733 / 6,725 / 6,819.
Re-derive these from the official dataset before final use.

## Headline evaluation
Prioritize:
- MUST-escalation recall;
- auto-handle coverage;
- unsafe-auto rate = share of auto-handled items humans label MUST or SHOULD.
Show a risk–coverage curve against always-escalate, never-escalate, and keyword rule.

Secondary:
- draft acceptability;
- intent macro-F1;
- historical behavior agreement, explicitly labelled as behavioral agreement.

## Baselines
At minimum include trivial baseline(s) such as majority/always-escalate/never-escalate/constant-DM and a simple baseline such as keyword rules, TF-IDF + Logistic Regression, or 1-NN.

## Generation
Use LLM only where drafting adds value. Cache outputs. Include a no-retrieval ablation if useful. For escalations, prefer deterministic handoff templates.

## Judge
Judge only what is necessary. Provisional rubric:
- addresses need;
- grounded in historical evidence;
- appropriate routing;
- tone;
- overall acceptability.
Validate against human ratings and deliberately corrupted replies. Freeze the rubric after one revision. Prefer a stable/pinned judge.

## Cost
Target $0–$10. Prefer free-tier/local generation and reserve paid spend, if any, for the validity-critical judge. Verify current pricing before spending.

## Main methodological risks
Retrieval leakage, near-duplicates, customer/thread overlap, unvalidated LLM judging, treating DM behavior as escalation truth, test-set threshold tuning, and biased stress sampling.
