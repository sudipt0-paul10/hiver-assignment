# Hiver SDE Intern — Working Instructions

Read `docs/assignment.md`, `docs/decisions.md`, and `docs/research.md` before making major decisions.
The taxonomy is locked in `docs/taxonomy.md` / `docs/codebook.json`; the pilot is written up in
`docs/pilot.md`, the golden evaluation set in `docs/golden_set.md`, and its completed
annotation results in `docs/golden_results.md`. Pipeline status is `docs/pipeline.md`,
baseline results `docs/results_baselines.md`, and the D12 retest `docs/retest.md`.

`assignment.md` summarizes the requirements; the official Hiver specification is the source of truth. `decisions.md` and `research.md` are prior proposals/evidence and MUST be validated against the actual dataset before final use.

## Principles
- Optimize evaluation quality and credibility over architectural complexity.
- Keep API spending extremely low: target ₹0–₹500, with an absolute ceiling of approximately ₹1,000, unless a clearly justified exception is approved.
- Prefer free/local methods for non-critical components and cache every LLM call.
- Do not add vector DBs, embeddings, GPUs, elaborate UI, or frameworks without evidence they improve the assignment.
- Keep the repo minimal, hygienic, and easy for the user to explain live.
- No duplicate experiment versions, dead scripts, abandoned notebooks, or unnecessary infrastructure.
- Keep generated artifacts in ignored output/cache directories.
- Never commit secrets; use `.env` locally.
- Cite borrowed datasets, papers, repositories, and methods.
- Never optimize human labels to make the model look good.
- Do not start the final golden-set labeling until the annotation pilot/codebook is reviewed and locked.

## Collaboration
Claude/Cowork is the implementation engineer. Inspect the actual code, data, and environment and make reasonable decisions autonomously.

For non-obvious changes, explain evidence, assumptions, and rationale. Do not invent prior work.

ChatGPT is an independent technical reviewer. Important architecture/evaluation decisions may be brought to ChatGPT for review before implementation.

## Current state
The repository started empty. Git hygiene has been established with `.gitignore` and `.gitattributes`. Build from the actual project state; do not assume an implementation already exists.
