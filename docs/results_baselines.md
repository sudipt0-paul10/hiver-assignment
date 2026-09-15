# Baseline results — intent and escalation

**Every number here comes from a real run.** Reproduce:

```bash
python scripts/classify.py --predict rule       --split golden
python scripts/classify.py --predict distant-lr --split golden
python scripts/policy.py   --all                --split golden
python scripts/evaluate.py --all --json outputs/eval/results.json
```

Full console output: `outputs/eval/results.txt`; machine-readable:
`outputs/eval/results.json`. Both are git-ignored and regenerable.

**No supervised baseline appears below, because none can exist (D23).** The 200
golden items are the only human labels in the project and they are TEST-window
items, so there is nothing to fit a supervised model on. Every row is labelled by
how it was constructed.

---

## 1. Intent

| Classifier | Construction | Core (n=160) | Full (n=200) | macro-F1 |
|---|---|---|---|---|
| `rule` | seed regex written on train; nothing fitted | **39.4% [32.1, 47.1]** | 47.5% [40.7, 54.4] | 0.474 |
| `distant-lr` | *proxy baseline* — TF-IDF+LR fitted on those seeds' output over train | **42.5% [35.1, 50.2]** | 50.0% [43.1, 56.9] | 0.520 |
| `llm` | zero-shot on the locked codebook | **not run** — no API key, needs D19 approval | | |

**Quote the core column.** Two gold items carry an intent outside the taxonomy
(G158, G179); no classifier can emit those values, so they count as errors.
Excluding them moves full accuracy by 0.5pp (47.5→48.0, 50.0→50.5) and changes
nothing.

### 1.1 Complexity has not earned its place yet

Paired McNemar, `distant-lr` vs `rule`: b=11, c=6, **p = 0.332**. The
distant-supervision model does **not** measurably beat the raw regex it was
trained on. On this evidence the LR adds nothing, which is exactly the test D15
demands before complexity is kept. It stays in the report as a baseline; it is
not a result.

### 1.2 The accuracy that must never be the headline

Accuracy by sampling-stratum family, `distant-lr`:

| Family | Accuracy | n |
|---|---|---:|
| `core` | 42.5% [35.1, 50.2] | 160 |
| `targeted` | 85.7% [60.1, 96.0] | 14 |
| `boundary` | 80.0% [49.0, 94.3] | 10 |
| `escalation` | 66.7% [39.1, 86.2] | 12 |
| `vague_unclear` | 100.0% [51.0, 100.0] | 4 |

**The `targeted` figure is circular.** Those 14 items were drawn *because* a seed
family matched them; `rule` is those same seeds and `distant-lr` is fitted on
their output. Both are being scored partly on items selected for agreeing with
them. 85.7% is an artefact of the sampling design, not performance. The
160-item core carries no such circularity, which is why it is the only number
quoted. The harness prints this warning next to the table so it cannot be lifted
out of context.

### 1.3 The dominant failure mode

Both classifiers collapse into `other_unclear`: they predict it for 114/200
(`rule`) and 104/200 (`distant-lr`) against a true rate of 30/200. Recall on that
class is 0.90–0.97 and precision 0.25–0.26 — it is absorbing everything.

Top confusions are the same for both, all in one direction:
`app_device_technical → other_unclear` (19), `account_access → other_unclear`
(13–15), `billing_subscription → other_unclear` (12–13).

This is the documented seed behaviour, not a new bug: `docs/taxonomy.md` §0
records that the keyword seeds left 60.2% of train in `other_unclear` and that
inspection showed most of that residue was assignable. The baselines inherit that
under-assignment wholesale. It is also the clearest argument for the LLM
classifier: the failure is vocabulary coverage, which is precisely what a
regex cannot fix and a language model might.

## 2. Escalation

Human base rate: **46.9% [39.3, 54.6]** on the core — the only population figure.
49.5% [42.6, 56.4] on all 200, inflated by the enriched strata.

D3 definitions, applied exactly: *coverage* = share of items auto-handled;
*unsafe-auto rate* = of those, the share the human labelled `ESCALATE`.

| Policy | Construction | Coverage | Unsafe-auto | Sees gold? |
|---|---|---:|---|---|
| `always-escalate` | constant | 0.0% | n/a | no |
| `lookup-codebook-rule` | codebook `escalation_role` on `rule` intent | 25.0% | 20.0% [11.2, 33.0] | no |
| `lookup-codebook-distant-lr` | same, on `distant-lr` intent | 29.5% | 22.0% [13.4, 34.1] | no |
| `distant-lr-score` | LR mass on `independent`-role intents, threshold 0.5 fixed a priori | 36.5% | 21.9% [14.0, 32.7] | no |
| `keyword` | MUST-proxy regexes ∪ DM-follow-up marker | 87.5% | 44.0% [36.9, 51.4] | no |
| `never-escalate` | constant | 100.0% | 49.5% [42.6, 56.4] | no |
| `lookup-oracle` | intent → majority gold decision, on **gold** intent | 51.0% | 6.9% [3.4, 13.5] | **YES — ceiling** |

`always-escalate` (coverage 0, trivially safe, useless) and `never-escalate`
(coverage 100, unsafe-auto = the base rate) are the two reference endpoints D5
requires.

### 2.1 Risk–coverage curve (`distant-lr-score`, the only scored policy)

| Coverage | Unsafe-auto | auto | unsafe |
|---:|---:|---:|---:|
| 10.0% | 5.0% | 20 | 1 |
| 25.0% | 16.0% | 50 | 8 |
| 50.0% | 24.0% | 100 | 24 |
| 75.0% | 36.0% | 150 | 54 |
| 90.0% | 43.9% | 180 | 79 |
| 100.0% | 49.5% | 200 | 99 |

### 2.2 Matched coverage — where the complexity question is settled

| Compared at | Baseline | `distant-lr-score` at that coverage |
|---|---|---|
| 25.0% | `lookup-codebook-rule` 10/50 = 20.0% | 8/50 = **16.0%** |
| 29.5% | `lookup-codebook-distant-lr` 13/59 = 22.0% | 12/59 = **20.3%** |
| 87.5% | `keyword` 77/175 = 44.0% | 74/175 = **42.3%** |
| 51.0% | `lookup-oracle` 7/102 = **6.9%** | 26/102 = 25.5% |

The scored policy is 1.7–4.0pp better than the lookup baselines at matched
coverage. That is a small edge on 50–175 items, and the paired tests agree it is
marginal: `distant-lr-score` vs `lookup-codebook-distant-lr` gives p = 0.057,
vs `lookup-codebook-rule` p = 0.035.

### 2.3 The gap that matters

Escalation accuracy: `lookup-oracle` **93.5% [89.2, 96.2]** against
`distant-lr-score` 70.0% [63.3, 75.9], `lookup-codebook-distant-lr` 66.0%,
`keyword` 60.0%. Every McNemar against the oracle is p < 0.0001.

The oracle reproduces the earlier in-sample figure exactly, which is the point:
**intent very nearly determines escalation under this policy.** So the whole
escalation problem reduces to intent classification, and the 23.5pp gap between
the best clean policy and the oracle is almost entirely intent error, not policy
error. Improving the escalation decision means improving the intent classifier.

`lookup-oracle` is fitted on the evaluation labels — twice over, since it uses
both gold intent and gold escalation. It is a ceiling, never a baseline, and is
labelled that way in the harness output, the prediction metadata and every table.

## 3. What is misleading about these numbers

1. **`targeted` accuracy is circular** (§1.2). The core is the honest number.
2. **The oracle is not achievable** — it is what perfect intent would buy.
3. **Unsafe-auto without coverage is meaningless.** `always-escalate` scores
   perfectly and is useless; the clean policies look good partly because they
   auto-handle only 25–37% of traffic.
4. **`distant-lr` is not supervised** and must never be reported as "TF-IDF+LR"
   without that qualifier. Its targets are regex output.
5. **No threshold was tuned.** The 0.5 cut is a priori because no labelled dev set
   exists. A tuned operating point would likely score better and could not be
   defended.
6. **One annotator, one pass.** The D12 retest (`docs/retest.md`) is drawn and due
   2026-09-19; until it runs, none of these numbers has a reliability ceiling
   attached.
7. **Two gold items sit outside the taxonomy**, unresolved, and count as errors.

## 4. The LLM classifier — implemented, not run

`scripts/classify.py --predict llm` is written and its prompt is inspectable with
`--show-prompt`. The prompt is assembled from `docs/codebook.json` alone:
intent definitions, confusable pairs and the boundary rules. No golden item
appears in it, and `--dev-sample N` exists so prompt work draws on dev text only.
Every call is cached under `cache/llm/` keyed on model, prompt and message, so a
second run is free.

It has not been run: there is no `ANTHROPIC_API_KEY` on this machine and the SDK
is not installed. Running it spends against the D19 budget and needs explicit
authorisation — the script refuses without `--allow-api` and says so.

---

## 5. Escalation reason codes (D9) — deterministic, nothing tuned

`scripts/policy.py` now attaches exactly one D9 reason to every `ESCALATE` and
none to any `AUTO_OK`. Selection is an ordered first-match-wins rule set over the
**predicted** intent and observable text only — MUST proxies, a PII pattern, the
DM-follow-up marker, the failed-self-service proxy, an anger pattern and an
out-of-scope pattern. Ties break by BR-9 (independent authority first). No golden
label is read, and no pattern or ordering was chosen by looking at how it scores.

### 5.1 Structural validity — the two required invariants

Checked by `python scripts/evaluate.py --reasons`, for all seven policies:

- **every `ESCALATE` carries exactly one valid reason** — 0 missing, 0 illegal,
  0 multi-valued, across 741 escalated rows in total;
- **every `AUTO_OK` carries no reason** — 0 stray reasons across 459 auto-handled
  rows.

### 5.2 Rule coverage — the honest weakness

| Policy | Decided by a text rule | By the intent fallback |
|---|---:|---:|
| `keyword` | 25/25 (100%) | 0 |
| `lookup-oracle` | 30/98 (30.6%) | 68 |
| `distant-lr-score` | 33/127 (26.0%) | 94 |
| `lookup-codebook-distant-lr` | 35/141 (24.8%) | 106 |
| `lookup-codebook-rule` | 35/150 (23.3%) | 115 |

Only about a quarter of reasons come from a text rule; the rest fall through to
the intent-derived fallback, and because the intent classifiers mostly predict
`other_unclear`, that fallback is overwhelmingly `ambiguous`. **The reason stage
is bottlenecked by intent quality exactly as the decision stage is** — the same
root cause, measured twice.

### 5.3 Agreement with the human reason (evaluation, not tuning)

On items where both the human and the policy escalate:

| Policy | Reason agreement |
|---|---|
| `lookup-oracle` (ceiling) | 73.9% [64.1, 81.8] n=92 |
| `keyword` | 72.7% [51.8, 86.8] n=22 |
| `distant-lr-score` | 53.0% [42.4, 63.4] n=83 |
| `lookup-codebook-distant-lr` | 52.3% [41.9, 62.6] n=86 |
| `lookup-codebook-rule` | 51.7% [41.5, 61.8] n=89 |

`keyword` scores well because it only escalates when a MUST proxy fired, so its
reason is nearly free — but it escalates just 25 items. The dominant error
everywhere else is the same: `human payment_dispute → predicted ambiguous` and
`human account_access → predicted ambiguous`.

### 5.4 Two reason-vocabulary coverage gaps, reported not patched

- **`public_pii`** — the human used it twice; the PII rule never fired on either.
  The pattern (email address, long digit run, "my email is …") does not match what
  the annotator saw as exposed personal data.
- **`ambiguous`** — the human used it 15 times; no *rule* produces it, only the
  fallback does. It is reachable, but never on evidence.

Both were found by the harness after the rules were frozen. Neither has been
patched: adjusting a rule now, having seen how it scores against the golden set,
is tuning on test. They are recorded for the report and for a future iteration
that has a labelled dev set to tune against.

## 6. Deterministic handoff (D8) — no generation on the escalation path

`scripts/handoff.py`. D8 fixes the requirement ("deterministic handoff template,
no generation"); the wording is ours and lives in one dictionary.

**The determinism guarantee is structural, not procedural.** `render(reason)` is a
pure lookup: it takes no customer text, interpolates nothing, and has exactly 11
possible outputs — one per D9 reason code. So a public reply on the escalation
path cannot echo back personal data the customer exposed, and the complete output
space can be printed and audited in one screen (`--selftest` does exactly that).

Guardrails, each checked on every rendered reply: no URL, no digit run of 7+, no
unfilled placeholder, ≤280 characters, no outcome or timeline promise (banned
phrase list), routes to a private channel, non-empty, and no 6-word window of the
customer's message present in the reply.

Results across all five escalating policies (541 rendered replies):

- **0 guardrail failures**
- **0 customer-text echoes**
- distinct replies per policy 6–10, bounded by the 11 templates as expected

Each row is auditable: `id, reason, template_id, reply, sha256, guardrails_passed`.

One guardrail bug was caught by the self-test rather than in production: the
private-channel check used `\bdm\b`, which did not match the `existing_case`
template's "check your DMs". Fixed in the regex, not by rewording the template.
