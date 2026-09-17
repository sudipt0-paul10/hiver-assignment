# D8 pipeline — implementation status

**Status updated 2026-09-17.** Sections 1–3 and 5 record the methodology as it was
decided and are unchanged. Sections 4, 6 and 7 record *status* and have been
brought up to date: the LLM stages have since run, on a local model, at zero cost.

Required architecture, quoted from `docs/decisions.md` D8 (unchanged, not reinvented):

```
customer message
  → text normalisation (D10)
  → intent classification
  → ESCALATE / AUTO_OK decision  (+ reason)
  → if AUTO_OK: retrieve similar historical cases → LLM draft → guardrails
  → if ESCALATE: deterministic handoff template (no generation)
```

| Stage | Status | Module |
|---|---|---|
| Corpus, threads, temporal split | **built** | `scripts/corpus.py` |
| Text normalisation (D10) | reused | `scripts/profile.py` |
| Retrieval + leakage controls | **built** | `scripts/retrieval.py` |
| Intent classification | **built and run** — rule, distant-lr, llm (`llama3.2:3b`) | `scripts/classify.py` |
| Escalation decision + reason | **built** | `scripts/policy.py` |
| LLM draft + guardrails | **built and run** — `llama3.2:3b`, 200 items | `scripts/draft.py` |
| Deterministic handoff template | **built** | `scripts/handoff.py` |
| Baselines (D14) | **built** | `scripts/classify.py`, `scripts/policy.py` |
| Evaluation harness (D3/D5/D7) | **built** | `scripts/evaluate.py` |
| LLM judge + corruption probes (D11) | **built and run** — `llama3.2:3b`, 200 judged + 80 probes | `scripts/judge.py` |

---

## 1. Corpus layer — built

`python scripts/corpus.py --build` streams `twcs.csv` once and caches the
SpotifyCares thread frame under `cache/corpus/` (git-ignored). Cached loads take
~2.8 s against ~31 s cold, and the cache carries a source fingerprint so a stale
copy is rebuilt rather than silently reused.

Measured, and matching `docs/profile.md` D18 exactly:

| | openers | window |
|---|---:|---|
| train | 14,547 | 2017-10-01 → 11-06 |
| dev | 6,716 | 2017-11-06 → 11-20 |
| test | 6,808 | 2017-11-20 → 12-04 |
| outside | 150 | — |

Threads straddling a split boundary: **0**, by construction. Retrieval corpus
(train openers with a brand reply and non-empty normalised text): **14,491**.

## 2. Retrieval layer — built

TF-IDF cosine (word 1–2 grams, `min_df=2`, sublinear tf) over D10-normalised
train openers, each carrying the brand's first reply. Per D8 retrieval *starts*
at TF-IDF; per D15 embeddings and a vector store are added only if a dev ablation
shows they earn their place.

**Leakage controls, and where each lives** — the point of the design is that
"we controlled for leakage" is a measurement here, not a claim.
`python scripts/retrieval.py --selftest` runs all 200 golden items and reports
what each guard removed:

| Control | Mechanism | Fired |
|---|---|---:|
| No self-retrieval | index is train-only; test tweet ids absent | 0 (structural) |
| No same-thread retrieval | index is train-only; test threads absent | 0 (structural) |
| Customer-disjoint | `exclude_customers` per query | **1 candidate, on G105** |
| Near-duplicate suppression | word-3-gram Jaccard ≥ 0.80, same definition as the sampler | 0 |

Two honest readings of that table:

- The **customer-disjointness guard is doing real work**: G105's own earlier
  train opener would otherwise have been retrieved. The self-test independently
  re-derives the 7 `customer_also_in_train` customers from the index and
  confirms the key's count.
- The **near-duplicate guard never fired** across 200 queries. It is currently
  insurance, not an active control, and should be reported that way rather than
  cited as evidence that near-duplicates were a threat we defeated.

Retrieval quality, measured: top-1 cosine median **0.310** (p10 0.219, p90 0.534,
min 0.168), which lands on D13's independently measured 0.303 median
opener-to-opener similarity. No item is left without evidence after filtering.
This is a weak retriever, exactly as D13 predicted, and that is the empirical
reason reply quality is judged (D11) rather than scored by reference overlap.

Results: `docs/results_baselines.md`.

## 3. Resolved: the label-availability decision (now D23)

**Human labels exist only on the test window.** The 200 golden items are the only
labelled data in the project. D7 forbids tuning on test, so a supervised intent
or escalation classifier has no labelled training set, and D14's "TF-IDF +
logistic regression" baseline has nothing to fit on.

Three ways out, with what each costs:

1. **Distant supervision from train.** Fit TF-IDF + LR on the seed-family weak
   labels over train openers (the same seeds used for sampling strata). Leak-free
   and cheap. It measures the seeds as much as the classifier — but that is a
   legitimate simple baseline, and D14 only asks the baseline to be simple.
2. **Zero/few-shot LLM classification** as the system, with the codebook in the
   prompt. Prompts are developed by inspection on **dev**, which needs no labels,
   and the golden set is scored once. This is the intended system per D8/D19.
3. **Cross-validation inside the golden set.** Rejected: fitting and reporting on
   the same 200 items is the contamination D7 exists to prevent.

**Adopted and recorded as D23** in `docs/decisions.md`: distant supervision for the
proxy baseline, zero/few-shot LLM for the system, cross-validation inside the
golden set never. D23 also states explicitly that the conventional supervised
TF-IDF+LR baseline named in D14 cannot be built at all.

A second consequence, already measured: the intent→escalation majority lookup
reproduces 93.5% of human escalation decisions in-sample
(`docs/golden_results.md` §7.1). That rule is now a **required** baseline — an
escalation model that has merely learned the intent must not be able to pass as a
safety result.

## 4. What has run, and what has not

**Run, on `llama3.2:3b` via a local OpenAI-compatible server, 0 billed calls:**

| Stage | Result | Artefact |
|---|---|---|
| LLM intent classifier | 200 items — core 38.1%, full 42.0%, macro-F1 0.333 | `outputs/preds/intent_llm_golden.csv` |
| LLM drafting + guardrails | 200 items — 43 auto-replies, 141 handoffs, 16 guardrail fallbacks | `outputs/preds/drafts_lookup-codebook-distant-lr_golden.csv` |
| LLM judge (D11) | 200/200 parsed, 86.0% acceptable | `outputs/preds/judge_lookup-codebook-distant-lr_golden.csv` |
| Corruption probes (D11) | 59/80 = 73.75% detected | `outputs/preds/probes_lookup-codebook-distant-lr_golden.csv` |

Every response is in `cache/llm/` and is committed, so these replay offline.
Full results and their caveats: `docs/report.md` §5–§7.

**Not run:**

1. **D11 human reply-quality ratings**, and therefore no judge–human agreement.
   The 60-row sheet in `data/golden/` holds AI-generated review suggestions, not
   human annotation, and is not used as a result. This is an open gap.
2. **D12 retest second pass** — pre-registered earliest start 2026-09-19, after
   the 09-17 deadline. The gate was not moved (`docs/retest.md` §5).

## 5. Reproducibility

`requirements.txt` now pins pandas, numpy, scikit-learn and openpyxl.
scikit-learn was newly installed for this stage (TF-IDF, and the LR baseline to
come); everything before it ran on pandas/numpy alone.

---

## 6. API budget (D19) — outcome: **₹0 / $0.00 spent**

**No billed API call was ever made.** The plan below was costed for the Anthropic
route and kept for the audit trail; it was superseded by the local
`openai-compat` route (D24), which cost nothing. Every stage metadata file
records `billed_calls: 0`, `usd: 0.0`. `scripts/classify.py --predict llm` still
refuses a billed provider without `--allow-api`, and no `ANTHROPIC_API_KEY` was
ever present.

### 6.1 The original costed plan (superseded, kept for the record)

**Token estimates are measured, not guessed**: the classifier prompt is 5,817
characters (~1,616 tokens) as built by `build_prompt()`; the mean golden message
is 127 characters (~35 tokens); mean 3-hit retrieval evidence is 632 characters
(~176 tokens), measured over 20 real queries at 3.6 chars/token.

| Stage | Calls | Input tok | Output tok |
|---|---:|---:|---:|
| Intent — 200 golden + 40 dev × 3 prompt iterations | 320 | 528k | 5k |
| Drafting — 200 golden + 20 dev × 3 iterations | 260 | 234k | 39k |
| Judge — 200 drafts + 80 corruption probes + 20 dev | 300 | 390k | 60k |
| **Total** | **880** | **1.15M** | **104k** |

Drafting covers all 200 so any policy's coverage can be scored without re-calling;
escalated items still receive the deterministic template, never a draft.

Costed at the published base rates (Haiku 4.5 $1/$5, Sonnet 5 $2/$10,
Opus 5 $5/$25 per MTok):

| Option | Generator + classifier | Judge | Cost | ×2 iteration margin |
|---|---|---|---:|---:|
| **A (recommended)** | Haiku 4.5 | Sonnet 5 | **≈ $2.36** | ≈ $4.72 |
| B | Sonnet 5 | Opus 5 | ≈ $5.41 | ≈ $10.82 |

At roughly ₹88/$: option A ≈ ₹208 (₹416 with margin), option B ≈ ₹476 (₹952 with
margin). D19 targets ₹0–500 with a ~₹1,000 ceiling, so **A sits inside the target
even after a doubling; B only fits if the prompts converge first time.** A also
gives a wider generator/judge tier gap, which is the only mitigation available for
D20's same-family judge bias.

**Caching**: every call is keyed on sha256(model, prompt, params, message) and
written to `cache/llm/`. Re-running a completed stage costs nothing; only an
edited prompt or a changed model invalidates entries. Whether that cache ships in
the repo for the <15-minute reproducibility requirement was D22; **it is now
resolved — the real-provider cache is committed** and the mock entries are not
(`.gitignore`: `!cache/llm/`, `cache/llm/*_mock_*`).


---

## 7. Provider abstraction — recorded as **D24** in `docs/decisions.md`

`scripts/providers.py` puts every model call behind one `complete(prompt,
message, params) -> Response` interface, with three implementations: `mock`
(default, offline, free), `openai-compat` (any self-hosted OpenAI-compatible
server, free) and `anthropic` (optional, billed, gated behind `--allow-api`).

**Why the default is a mock rather than a local model.** A local model is the
better default in principle, and `openai-compat` is the supported path to one. It
could not be the default on the machine the refactor was written on: 2 cores,
3.9 GB RAM, no GPU, and an egress allowlist blocking `huggingface.co` and
`ollama.com` (measured — `pypi.org` answers 200, both model hosts answer 000).
The runtime installs; the weights could not be fetched. So the offline default is
a stub, and the free path to real output is a server run on a machine that can
reach weights.

**That path is what shipped.** The LLM stages were subsequently run against
Ollama serving `llama3.2:3b` on a machine that could fetch the weights, via
`--provider openai-compat --base-url http://127.0.0.1:11434/v1`. The mock
remains the zero-dependency default; it is no longer the only thing that has
run.

**Three barriers keep mock output from being mistaken for a result:** every
response is stamped `is_mock`, every artefact gets a `_mock` filename suffix and
`is_mock: true` in its metadata, and `evaluate.py` prints mock rows under a
`[MOCK PROVIDER - NOT A RESULT]` banner and excludes them from the baseline
comparison and from McNemar. The cache key includes the provider, so a mock entry
and a real entry can never collide or silently substitute.

**What the refactor bought, and what is still missing from deliverable 3.** The
one-flag switch let a real judge run at zero cost, and its responses ship in the
cache so a grader reproduces them with no credentials and no model. Deliverable 3
also asks for judge–human agreement evidence, and that half is **not** delivered:
see `docs/report.md` §8. The judge is qualified here by its corruption probes
alone.
