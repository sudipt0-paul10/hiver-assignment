# Golden set — annotation results and characterisation

**Status: 200/200 annotated, validated, and characterised. No system exists yet,
so nothing here is a system evaluation.** Codebook **0.2.1-proposed**, locked.

Reproduce:

```bash
python scripts/sample_golden.py --export                                    # xlsx -> annotated CSV, gated on text integrity
python scripts/sample_golden.py --verify                                    # the pristine sample is unaltered
python scripts/sample_golden.py --validate data/golden/golden_annotated.csv # schema + pilot diagnostics
python scripts/golden_report.py --json outputs/golden/golden_report.json    # this document's numbers
```

| File | Role |
|---|---|
| `data/golden/golden_items.xlsx` | **authoritative human annotation**, 200 rows |
| `data/golden/golden_annotated.csv` | machine-readable export of the above, UTF-8 no BOM |
| `data/golden/golden_items.csv` | the pristine blank sample — unchanged, the integrity anchor |
| `data/golden/golden_key.json` | sampling metadata — unchanged |
| `outputs/golden/golden_report.{txt,json}` | regenerable report (git-ignored) |

---

## 1. What can and cannot be computed at this stage

**There are no predictions.** No classifier, retriever, generator or judge has
been built, so there is no accuracy, no F1, no precision/recall, and no
prediction-vs-gold confusion matrix. Any such number quoted now would be
fabricated.

**`golden_key.json` is not a label key.** It holds stratum, tweet/thread/customer
id, inclusion probability, design weight and the `customer_also_in_train` flag —
no reference intents and no reference escalation decisions. The human labels are
the reference; there is nothing above them to score them against. The only
"disagreement" that exists is **seed-proxy vs human label** (§6), which is a
boundary diagnostic and explicitly not an error signal.

What follows is therefore a *characterisation of the labels*: distributions with
Wilson 95% intervals, split core / targeted / full, plus the ambiguity register
that determines what must be settled before a system is built on this set.

Every core item shares one inclusion probability (0.0246), so the
design-weighted and unweighted core estimates are identical. Population
statements use **core only**; the enriched 40 are reported separately and never
enter a population number.

## 2. Validation status

| Check | Result |
|---|---|
| Row count | 200 |
| Ids unique, and identical to the sampled set | pass |
| Tweet text vs the pristine sample | 200/200 byte-identical |
| Tweet text vs `twcs.csv` | 200/200 byte-identical |
| Encoding drift from the spreadsheet round-trip | none |
| Thread ids / customer ids | untouched (they were never in the annotation file) |
| One item per thread, customer-disjoint | 200 threads, 200 customers |
| Every annotation field populated | yes |
| Values conform to the taxonomy | **2 violations — see §3** |
| `AUTO_OK` ⇒ reason `none`/blank; `ESCALATE` ⇒ reason ≠ `none` | pass |

The xlsx→CSV export is gated: it refuses to write if the row count, id set or
any text differs from either the pristine sample or `twcs.csv`. The mojibake
failure that hit the pilot (10 of 40 rows) did **not** recur — the spreadsheet
surface worked as intended.

## 3. Two rows outside the taxonomy — open, unresolved

| id | stratum | intent as annotated | text |
|---|---|---|---|
| G158 | `escalation:legal_rights` | `legal_rights` | *"Why is there no Disconnect from Facebook button, why do we need to 'log a ticket'. Good luck with GDPR."* |
| G179 | `core` | `legal_rights` | *"This guy named Young Summit re-posted @23568's song … Josh A's song was made in 2015 …"* |

`legal_rights` is an **escalation reason**, not one of the eight intents. Both
rows are reported as `INVALID` in every table below and **were not changed**.

They are not typos. Both are cases the locked codebook has no class for: a data-
rights/GDPR complaint and a creator-side copyright dispute — the latter is the
artist/music-distribution gap already recorded in `docs/taxonomy.md` §6. The
codebook's prescribed handling is `other_unclear` + `cannot_represent: y`;
`cannot_represent` was used **zero times** across all 200 items, so the escape
hatch that exists precisely for this was not exercised.

**This is the owner's call, not the annotator's and not the engineer's**, because
it decides §4's verdict.

## 4. Intent — and the falsification threshold

| Intent | Core (n=160) | Targeted (n=40) | Full (n=200) |
|---|---|---|---|
| `app_device_technical` | 30 — 18.8% [13.5, 25.5] | 4 — 10.0% | 34 — 17.0% [12.4, 22.8] |
| `billing_subscription` | 27 — 16.9% [11.9, 23.4] | 6 — 15.0% | 33 — 16.5% [12.0, 22.3] |
| `other_unclear` | 24 — 15.0% [10.3, 21.3] | 6 — 15.0% | 30 — 15.0% [10.7, 20.6] |
| `account_access` | 23 — 14.4% [9.8, 20.6] | 10 — 25.0% | 33 — 16.5% [12.0, 22.3] |
| `product_feature_feedback` | 16 — 10.0% [6.2, 15.6] | 4 — 10.0% | 20 — 10.0% [6.6, 14.9] |
| `playback_playlist` | 14 — 8.8% [5.3, 14.2] | 4 — 10.0% | 18 — 9.0% [5.8, 13.8] |
| `plans_eligibility` | 13 — 8.1% [4.8, 13.4] | 4 — 10.0% | 17 — 8.5% [5.4, 13.2] |
| `content_availability` | 12 — 7.5% [4.3, 12.7] | 1 — 2.5% | 13 — 6.5% [3.8, 10.8] |
| `INVALID: legal_rights` | 1 | 1 | 2 — 1.0% |

**Prevalence vs the taxonomy's own estimates.** The core column is the first real
measurement of prevalence; everything in `docs/taxonomy.md` §2 was a proxy range.
`playback_playlist` was projected at **20–30%** and measured **8.8%
[5.3, 14.2]** — the projection sits far outside the interval, and the class the
codebook called "the largest" and "the volume class where auto-handle coverage
is won" is in fact one of the smaller ones. `account_access` (10–15% → 14.4%),
`billing_subscription` (6–9% → 16.9%, above range), `app_device_technical`
(10–14% → 18.8%, above range), `content_availability` (5–8% → 7.5%),
`plans_eligibility` (5–8% → 8.1%) and `product_feature_feedback` (4–8% → 10.0%)
are closer. The NMF/keyword proxies over-weighted playback and under-weighted the
account/billing/device families. **This should be corrected in the report rather
than left standing**, and it changes where auto-handle coverage actually comes
from.

**`other_unclear` = 15.0%** in core and full alike. `docs/taxonomy.md` §7 sets
the falsification criterion at **">15% ⇒ the taxonomy is inadequate"**:

- **as annotated:** core 15.0% [10.3, 21.3], full 15.0% [10.7, 20.6] — exactly on
  the line, not over it;
- **if G158 and G179 resolve to `other_unclear`** per the codebook: core 15.6%
  [10.8, 22.0], full 16.0% [11.6, 21.7] — **over the line**.

So the verdict on the locked taxonomy turns on two rows. Reported, not resolved.

For context, the pilot put 7.5% (3/40) here; the interval on 3/40 was wide enough
to contain 15%, so this is not a contradiction — it is the first estimate with
enough denominator to be worth anything.

**One evidence pattern worth having in front of you when you rule.** All 15
low-confidence items are `other_unclear`, and 10 of 10 boundary-stratum items
landed *inside* the pair their proxy predicted (§6), 6 of them at high
confidence. The annotator's hesitation is concentrated entirely in "the text
does not say what the problem is", not at the class boundaries the codebook
defines. Three of the unclear items (G047, G081, G191) explicitly depend on a linked
image the annotator could not see, and several more carry a bare URL as their
only content. Whether that makes the taxonomy inadequate or the tweets
uninformative is the judgement to make; the pattern is stated here without a
recommendation.

## 5. Attributes

| Attribute | Core | Full | Note |
|---|---|---|---|
| `conversation_state` = `opener` | 96.2% | 95.5% | **near-constant** |
| `existing_case_followup` | 3.8% | 4.5% | 9 items; `unclear` never used |
| `urgency` low/normal/high | 30.0 / 30.6 / 39.4% | 27.0 / 32.0 / 41.0% | informative |
| `frustration` low/normal/high | 55.6 / 30.0 / 14.4% | 52.5 / 32.0 / 15.5% | informative |
| `language` = `english` | 95.0% | 95.5% | **near-constant**; 8 non-English, 1 unclear |
| `confidence` low/medium/high | 6.9 / 34.4 / 58.8% | 7.5 / 34.5 / 58.0% | |

`conversation_state` and `language` are near-constant and will not discriminate:
report them descriptively, do not build a metric whose denominator is 9 items.
`urgency` and `frustration` do carry variation. Note that `existing_case_followup`
at 4.5% is **ten times** the 0.41% measured over train openers in
`docs/taxonomy.md` §1 — the enriched `escalation:existing_case` stratum explains
part of it, but core alone is 3.8%, so the original figure looks like an
undercount by the DM-language regex.

## 6. Seed proxy vs human label

Strata are retrieval devices; disagreement is a boundary signal, not an error.

- **Targeted strata: 12/14 agreed.** G106 (seeded `content_availability` →
  labelled `product_feature_feedback`) and G063 (seeded
  `product_feature_feedback` → labelled `plans_eligibility`, high confidence).
- **Boundary pairs: 10/10 landed inside the seeded pair.** The BR-2/3/4/6/11
  boundaries the pilot exposed did not produce confusion here.
- **Escalation proxies: 11/12 escalated.** The one that did not is
  `escalation:safety_abuse` G087 — the antivirus tweet flagged at sampling time
  as a proxy false positive, confirmed as `AUTO_OK` by the annotator.
- **Reason codes matched the proxy 7/12.** `escalation:failed_self_service` drew
  two items the annotator escalated for `security` instead; `escalation:security`
  and `escalation:existing_case` each drew one re-reasoned item. The proxies find
  escalation-positive items reliably and predict the *reason* poorly.
- **`vague_unclear`: 4/4 → `other_unclear`, all ESCALATE, all low confidence.**

## 7. Escalation

| Subset | ESCALATE | Wilson 95% |
|---|---|---|
| Core (unbiased) | 75/160 — **46.9%** | [39.3, 54.6] |
| Targeted (enriched) | 24/40 — 60.0% | [44.6, 73.7] |
| Full | 99/200 — 49.5% | [42.6, 56.4] |

**46.9% is the base rate to quote.** The targeted rate is inflated by
construction. The pilot's 35% (14/40, Wilson [22.1, 50.5]) is lower, but the two intervals
overlap over [39.3, 50.5], so the estimates are consistent rather than in
conflict — 40 items simply could not resolve this.

Reason distribution, full set: `payment_dispute` 32, `account_access` 24,
`ambiguous` 15, `out_of_scope` 8, `security` 7, `failed_self_service` 5,
`existing_case` 3, `public_pii` 2, `legal_rights` 2, `high_frustration` 1,
**`safety_abuse` 0**.

- **`safety_abuse` is an empty cell.** As predicted at sampling time (4 eligible
  items in a 6,495 pool). Report it as *not measurable*, never as zero error.
- `high_frustration` (1) and `public_pii` (2) are effectively unmeasurable too.
- `public_pii` occurring at all vindicates carrying the **exact original text**:
  D10-normalised text deletes the mentions and URLs those two labels rest on.

### 7.1 The escalation headline's biggest threat

| Intent | ESCALATE / AUTO_OK | |
|---|---|---|
| `billing_subscription` | 33 / 0 | **pure** |
| `product_feature_feedback` | 0 / 20 | **pure** |
| `content_availability` | 0 / 13 | **pure** |
| `account_access` | 32 / 1 | near-pure |
| `app_device_technical` | 4 / 30 | near-pure |
| `plans_eligibility` | 1 / 16 | near-pure |
| `playback_playlist` | 2 / 16 | near-pure |
| `other_unclear` | 25 / 5 | mixed |

**A lookup table mapping each intent to its majority escalation decision
reproduces 187/200 = 93.5% of the human decisions.** This is an in-sample ceiling
fitted on these labels and will score lower out of sample, but the implication
stands: escalation is very nearly a deterministic function of intent under this
policy.

Consequences, all of which belong in the report:

1. **This lookup rule is a mandatory baseline.** Without it, an escalation
   classifier that has merely learned the intent will look like a safety result.
   It is the "trivial baseline" deliverable 4 asks for, and it is a strong one.
2. It is **policy-consistent, not an annotation defect** — the codebook grants
   `account_access`, `billing_subscription` and `other_unclear` independent
   escalation authority, so purity is the policy working as written.
3. It narrows what the escalation metric measures to the residual: the 13 items
   the lookup gets wrong, and `other_unclear`'s genuine 25/5 split.
4. It is a direct answer to **"what is misleading about my headline number?"**

## 8. Leakage and isolation

`customer_also_in_train`: **7 items** — G027, G047, G067, G090, G105, G130, G179.
**Retained**, per `docs/golden_set.md` §2: dropping them would bias the set
against repeat customers and flatter retrieval. The obligation lands on the
retrieval stage, which must exclude same-customer and same-thread evidence for
these ids. Nothing about them changed during annotation.

Thread-level isolation (200 distinct threads) and customer disjointness within
the set (200 distinct customers) are re-verified above.

## 9. Intra-annotator reliability — required, not optional

**D12 is already project policy**: ~30 test–retest items, blind, after a
deliberate delay. It is not a nice-to-have:

- Deliverable 3 requires **evidence of judge–human agreement**. With one
  annotator and no self-agreement figure, "the judge agrees with the human at
  κ = 0.62" is uninterpretable — it could be a poor judge or a noisy task. Self-
  agreement is the ceiling that makes it readable.
- This set has two live reliability questions of its own: a 15.0% `other_unclear`
  rate sitting on the falsification line, and 15 low-confidence items all in that
  one class. A retest measures exactly that instability.

**Drawn and documented: `docs/retest.md`** — 30 items, seed 1212, verified 9/9,
earliest start 2026-09-19. **That date falls after the 2026-09-17 submission
deadline and was deliberately not moved**, so the second pass did not run and no
reliability ceiling exists at submission (`docs/retest.md` §5). What follows was
the recommendation and is now the procedure.

**Recommendation: do it, and start the clock now.** Draw the ~30 items (stratified
to over-sample `other_unclear` and the low-confidence rows), then re-label after a
real delay — several days, not hours — while system building proceeds in parallel.
It costs roughly 30 minutes of labelling and converts a headline caveat into a
measured number. It does **not** block building the agent.

## 10. Limitations carried forward

- **No system-vs-human metric exists yet.** Everything above is a description of
  the labels.
- **One annotator, one pass, no second rater.** Unchanged and still the largest
  threat to every number derived from this set. §9 partially addresses it.
- **`safety_abuse`, `high_frustration`, `public_pii` are unmeasurable** at n=0/1/2.
- **Two items sit outside the taxonomy** (§3), unresolved.
- **The enriched 40 are not a population sample** and never enter a headline.
- **The pool is restricted to threads that received a brand reply**, and one item
  per thread/customer removes 261 repeat-customer openers.
- **At least 3 items are unanswerable from text alone** (G047, G081, G191) —
  they depend on a linked image.
  That is a hard ceiling on any text-only system and should be stated as such
  rather than counted as model error.
- **The prevalence proxies in `docs/taxonomy.md` §2 are now known to be wrong**
  for `playback_playlist` in particular (§4). The taxonomy document should carry
  a pointer to the measured figures; its estimates must not be quoted as results.
