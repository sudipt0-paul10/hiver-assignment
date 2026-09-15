# D8 pipeline — implementation status and plan

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
| Intent classification | **built** (rule, distant-lr; llm written, not run) | `scripts/classify.py` |
| Escalation decision + reason | **built** | `scripts/policy.py` |
| LLM draft + guardrails | **built**, dry-run tested, unrun | `scripts/draft.py` |
| Deterministic handoff template | **built** | `scripts/handoff.py` |
| Baselines (D14) | **built** | `scripts/classify.py`, `scripts/policy.py` |
| Evaluation harness (D3/D5/D7) | **built** | `scripts/evaluate.py` |
| LLM judge + corruption probes (D11) | **built**, selftested, unrun | `scripts/judge.py` |

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

## 4. Next steps, in order

1. **LLM intent classifier run** — written and cached-by-design; needs an API key
   and explicit D19 budget approval.
2. **LLM drafting + guardrails**, then the judge and its validation (D11).
3. **D12 retest** on/after 2026-09-19, which supplies the reliability ceiling
   every judge–human agreement figure has to be read against.

Every remaining no-API component is built. Steps 1–2 need an API key and explicit
D19 authorisation; step 3 needs neither.

## 5. Reproducibility

`requirements.txt` now pins pandas, numpy, scikit-learn and openpyxl.
scikit-learn was newly installed for this stage (TF-IDF, and the LR baseline to
come); everything before it ran on pandas/numpy alone.

---

## 6. API budget plan (D19) — awaiting authorisation, nothing spent

No API call has been made. `scripts/classify.py --predict llm` refuses without
`--allow-api`, and there is no `ANTHROPIC_API_KEY` on this machine.

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
the repo for the <15-minute reproducibility requirement is D22, still open.
