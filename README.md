# AI support agent on TWCS — SpotifyCares

Hiver SDE intern take-home. An evaluation-first support agent over the Customer
Support on Twitter dataset: it classifies an incoming customer message into one
of eight intents, decides auto-handle vs escalate with a stated reason, and
either drafts a reply grounded in how Spotify historically answered similar
cases or hands off with a deterministic template.

The emphasis is on proving whether it works. Most of this repository is
measurement: a 200-item hand-labelled golden set, a leakage-controlled retrieval
layer, four escalation baselines including a mandatory trivial one, and an
evaluation harness that reports Wilson intervals and paired McNemar tests.

---

## 1. Get the dataset

**The raw dataset is NOT committed.** `twcs.csv` is ~493 MB, above GitHub's
100 MB hard limit, and it is a public Kaggle dataset rather than our work.

| | |
|---|---|
| Source | Kaggle: `thoughtvector/customer-support-on-twitter` |
| URL | https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter |
| File needed | `twcs.csv` (inside the Kaggle archive, under `twcs/`) |
| Expected path | `twcs/twcs.csv`, relative to the repository root |
| Size | 516,508,641 bytes |
| Lines | 3,002,524 (1 header + 3,002,523 rows) |
| SHA-256 | `cd297fcfa1bf6f99938be242e8e578980bc6d1b96adc8691abec9a39175b03c0` |

```bash
# with the Kaggle CLI, from the repository root
kaggle datasets download -d thoughtvector/customer-support-on-twitter
unzip customer-support-on-twitter.zip          # yields twcs/twcs.csv
sha256sum twcs/twcs.csv                        # must match the hash above
```

Or download the archive from the URL above and place `twcs.csv` at
`twcs/twcs.csv` yourself. Every script defaults to that path and accepts
`--csv` to override it.

Nothing else needs downloading: the golden set, the codebook and all sampling
artefacts are committed.

## 2. Install

Python 3.10 (developed on 3.10.12).

```bash
pip install -r requirements.txt
```

## 3. Reproduce

**Every result in this repository was produced with `$0.00` of API spend.** The
LLM stages ran on a local `llama3.2:3b` server, and **their responses are
committed** under `cache/llm/`. A grader with no model, no key and no network
replays them and gets the same numbers.

### 3.1 The headline results

```bash
python scripts/corpus.py --build            # one streaming pass, ~30 s, caches locally

# baselines - no API key, no network
python scripts/classify.py --predict rule       --split golden
python scripts/classify.py --predict distant-lr --split golden
python scripts/policy.py   --all --split golden
python scripts/handoff.py  --render lookup-codebook-distant-lr --split golden

# the single command that produces every table in docs/report.md
python scripts/evaluate.py --all --json outputs/eval/results.json
```

`evaluate.py` reads only the prediction CSVs already in `outputs/preds/` plus the
golden set. It makes no model call and needs no provider, so this step is offline
and unconditional.

### 3.2 Headline metrics this should print

| | |
|---|---|
| Intent, core n=160 | `distant-lr` 42.5% · `rule` 39.4% · `llm` 38.1% |
| Intent, macro-F1 full | `distant-lr` 0.520 · `rule` 0.474 · `llm` 0.333 |
| **D3 headline** — `lookup-codebook-distant-lr` | 29.5% coverage, **22.0%** unsafe-auto |
| **D3 end-to-end**, after drafting + guardrails | 21.5% coverage, **27.9%** unsafe-auto |
| Judge (`llama3.2:3b`) | 200/200 parsed, 86.0% acceptable |
| Corruption probes (D11) | 59/80 = 73.75% detected |

Read these against `docs/report.md` §7 before quoting any of them. The 86% judge
figure and the 80% targeted-intent figure are both misleading, for reasons
documented there.

### 3.3 Re-running the LLM stages — optional, not needed for §3.2

All three LLM stages are already run and their outputs are in `outputs/preds/`.
To regenerate them, the committed cache is replayed before any call, so this
costs nothing and contacts no server:

```bash
python scripts/classify.py --predict llm --split golden \
       --provider openai-compat --base-url http://127.0.0.1:11434/v1 --model llama3.2:3b
python scripts/draft.py  --policy lookup-codebook-distant-lr \
       --provider openai-compat --base-url http://127.0.0.1:11434/v1 --model llama3.2:3b
python scripts/judge.py  --policy lookup-codebook-distant-lr \
       --provider openai-compat --base-url http://127.0.0.1:11434/v1 --model llama3.2:3b
python scripts/judge.py  --probes --policy lookup-codebook-distant-lr \
       --provider openai-compat --base-url http://127.0.0.1:11434/v1 --model llama3.2:3b
```

**Pass `--provider openai-compat`.** Omitting it selects the default `mock`
provider, which writes stub output to `_mock`-suffixed files — useful for testing
the wiring, never a result. A cache *miss* (an edited prompt, a different model)
falls through to the server at `--base-url`; to cover that case, run `ollama
serve` with `llama3.2:3b` pulled. With the cache intact, no server is contacted.

Verification, also key-free:

```bash
python scripts/sample_golden.py  --verify     # 13 golden-set integrity checks
python scripts/retest_sample.py  --verify     # 9 retest checks + the delay gate
python scripts/retrieval.py      --selftest   # leakage guards, measured not asserted
python scripts/handoff.py        --selftest
python scripts/draft.py          --selftest
python scripts/judge.py          --selftest
```

### 3.4 Providers

The provider is pluggable (`scripts/providers.py`, D24):

| `--provider` | Cost | Needs | Output |
|---|---|---|---|
| `mock` *(default)* | free | nothing | deterministic stub — **never a result** |
| `openai-compat` | free | a self-hosted server (`--base-url`) | real model output — **what shipped** |
| `anthropic` | billed | `ANTHROPIC_API_KEY` in `.env`, plus `--allow-api` | real model output — **never used; $0.00 spent** |

**The mock does not simulate a model.** It exercises prompt assembly, retrieval
wiring, guardrails, routing, caching and artefact schemas so the pipeline is
runnable and testable end to end with no dependencies. Its artefacts are written
with a `_mock` filename suffix and `is_mock: true`, and `evaluate.py` prints them
under a `[MOCK PROVIDER — NOT A RESULT]` banner and excludes them from the
McNemar comparison. Numbers from a mock run measure the harness, not the system.

Any OpenAI-compatible server works (Ollama, `llama-server`, LM Studio, vLLM).
What shipped here: Ollama on `http://127.0.0.1:11434/v1` serving `llama3.2:3b`,
for the intent classifier, the drafter and the judge alike. That the generator
and the judge are the *same* model is a recorded limitation — see D20 and
`docs/report.md` §7.

## 4. What is committed, and what is not

**Committed** — scripts, documentation, the golden set and its sampling
artefacts, the codebook, `requirements.txt`, the real-provider LLM response
cache, and the report as both `docs/report.md` and `docs/report.pdf`.

**Not committed**, all regenerable or secret:

| Path | Why |
|---|---|
| `twcs/` | 493 MB public dataset — fetch per §1 |
| `.env` | API key |
| `cache/corpus/` | 14 MB pickle, rebuilt from `twcs.csv` in ~30 s |
| `outputs/` | every metric and prediction, regenerated by §3 |
| `__pycache__/`, `.ruff_cache/` | tooling caches |

`outputs/` is git-ignored, so the rendered report is committed at
**`docs/report.pdf`**, not at `outputs/report.pdf`. `outputs/eval/results.txt`
and `results.json` are the harness's own record of the run and are regenerated by
the last command in §3.1.

## 5. Layout

```
scripts/        pipeline and evaluation (see docs/pipeline.md)
data/golden/    200-item golden set, sampling key, manifests, D12 retest
docs/           taxonomy, codebook, decision log, sampling notes, results
cache/llm/      committed LLM response cache (replayed, never re-billed)
outputs/        generated; git-ignored
```

## 6. Documentation

| Document | Contents |
|---|---|
| `docs/report.md` | **the submission report** — framing, results, failure modes, limitations |
| `docs/decisions.md` | the decision log, D1–D24, with the 13-decision submission subset marked |
| `docs/taxonomy.md` | intent codebook 0.2.1, boundary rules BR-1…BR-11 |
| `docs/golden_set.md` | golden-set sampling methodology |
| `docs/golden_results.md` | what the 200 annotations actually show |
| `docs/results_baselines.md` | intent and escalation results vs baselines |
| `docs/pipeline.md` | D8 pipeline status and API budget |
| `docs/retest.md` | D12 intra-annotator retest protocol |
| `docs/pilot.md` | the 40-item codebook pilot |
| `docs/profile.md` | dataset profiling that drove brand selection |

## 7. Status at submission (2026-09-17)

**Complete.** Brand selection; taxonomy and codebook; the 200-item golden set,
annotated and verified 13/13; retrieval with measured leakage controls;
escalation policy and reason codes; deterministic handoff; the `rule` and
`distant-lr` baselines; the **LLM intent classifier, drafter and judge, all run
on real local `llama3.2:3b` inference**; the D11 corruption probes; the full
evaluation harness; and the report (`docs/report.md`).

**Deliberately not done.**

- **The D12 retest second pass.** The artefact is drawn and verified 9/9, but its
  pre-registered earliest start is **2026-09-19**, two days after this deadline.
  The gate was set on 09-14 and was **not moved to fit the deadline**. See
  `docs/retest.md` §5.

**Not done — a genuine gap.**

- **D11's human-agreement half.** No judge–human agreement figure is reported.
  `data/golden/human_ratings_lookup-codebook-distant-lr.csv` contains 60 rows of
  **AI-generated review suggestions, not human annotation**; they are retained
  unaltered as an artefact of the attempt and are not used as a result anywhere.
  D11 is validated by its corruption probes only. `docs/report.md` §8 states the
  limitation and the three instrument faults found along the way.
