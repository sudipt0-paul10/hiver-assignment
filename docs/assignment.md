# Hiver SDE Intern Take-Home — Assignment Summary

Official specification:
https://docs.google.com/document/d/18xREErNCLCGoH5fDvmciwrq7vuAPsJ-1nnt6LMtlJZ4/mobilebasic#heading=h.ofvgr7a884wk

This is a working summary, not a replacement for the official specification.

## Goal
Using the Customer Support on Twitter (TWCS) dataset, build an AI support agent for one selected brand that:
1. classifies an incoming customer message into a small set of intents derived from the data;
2. drafts a reply grounded in how the brand historically resolved similar issues;
3. decides auto-handle vs escalate, with a stated reason.

The assignment emphasizes proof/evaluation over system complexity.

## Data
Primary: TWCS (`twcs.csv`), commonly distributed as Kaggle `thoughtvector/customer-support-on-twitter`.
Optional: Banking77 for intent work if useful.
A full-dataset run is not required; a defensible subsample is acceptable.

## Deliverables
1. Runnable repo; README reproduces headline results in <15 minutes.
2. 150–250 human-labelled golden examples plus sampling/labeling note.
3. Evaluation harness with automated metrics, LLM-as-judge rubric, and evidence of judge-human agreement.
4. Report <=6 pages (or README equivalent) covering problem framing, results vs at least two baselines (trivial + simple), five failure modes with real examples/hypotheses, “What is misleading about my headline number?”, and what one more week would change.
5. Decision log of 10–15 non-obvious decisions and rationale.
6. AI coding assistants are allowed; live explanation/modification is expected.

## Methodological requirements
Prevent leakage and misleading evaluation:
- a test message must not retrieve/copy its own historical response;
- control customer/thread overlap and near-duplicates;
- prefer a justified temporal split;
- distinguish human policy labels from historical brand behavior;
- do not claim TWCS contains true escalation ground truth.

## Cost constraint
Keep API spend very low, target $0–$10, cache calls, and avoid unnecessary bulk LLM processing or paid infrastructure.
