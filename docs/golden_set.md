# Golden evaluation set — sampling and labelling note

**Status: sampled, not yet annotated.** 200 items, all annotation fields blank.
Codebook **0.2.1-proposed**, locked (`docs/codebook.json`, `docs/taxonomy.md`).

Reproduce end to end:

```bash
python scripts/sample_golden.py --build      # writes data/golden/*
python scripts/sample_golden.py --verify     # 13 structural checks against twcs.csv
python scripts/sample_golden.py --validate data/golden/golden_items.csv   # after annotation
```

`--build` is deterministic: a rebuild reproduces `golden_items.csv` and
`golden_key.json` byte-for-byte (verified by SHA-256).

| File | Contents | Annotator may open |
|---|---|---|
| `data/golden/golden_items.csv` | the 200 items, canonical, UTF-8 no BOM | yes |
| `data/golden/golden_items.xlsx` | the same 200 items as a spreadsheet, with dropdowns | yes — **annotate here** |
| `data/golden/golden_key.json` | strata, ids, inclusion probabilities, design weights | **no** |
| `data/golden/golden_manifest.json` | full sampling record | **no** |

`data/golden/` is tracked by git (the `.gitignore` carve-out exists for exactly
this). The pilot's warning applies in reverse: the golden annotations are
irreplaceable human work and must not live in an ignored directory.

---

## 1. Split, and why there is no `--split` flag

**TEST only: 2017-11-20 → 2017-12-04**, on the thread-opener date, per D18.
Whole threads follow their opener, so boundary-straddling is structurally
impossible (measured 0).

`scripts/annotate_pilot.py` has a `--split` flag because the pilot deliberately
used **dev**: its output was a codebook revision, and revising a codebook
against test items is tuning against the evaluation set (D7). The golden set is
the evaluation set, so there is nothing to choose — `SPLIT` is a module constant,
not a flag.

Pool universe: SpotifyCares customer openers in the test window **whose thread
received a brand reply**. The brand-reply requirement is inherited from the
pilot. It is not needed to label intent or escalation — those come from the
customer message alone — but it keeps a historical reply available for the
behavioural-agreement analysis (D9/D16) and for reply-quality work later. It is
a real restriction on the population and is recorded as such in §8.

## 2. Filtering, with counts

| Step | Removed | Remaining |
|---|---:|---:|
| Test-window openers with a brand reply | — | 6,808 |
| Empty after D10 normalisation | 32 | 6,776 |
| In the pilot's excluded **threads** | 0 | 6,776 |
| In the pilot's excluded **customers** | 0 | 6,776 |
| In the pilot's excluded **tweets** | 0 | 6,776 |
| Second+ opener of a thread | 0 | 6,776 |
| Repeat customer (customer-disjoint) | 261 | 6,515 |
| Exact normalised-text duplicate | 16 | 6,499 |
| Near-duplicate of another test item | 4 | 6,495 |
| Near-duplicate of a pilot item | 0 | **6,495** |

All three pilot-exclusion counts are 0, and that is expected rather than
suspicious: the pilot was drawn from **dev**, so no pilot thread or tweet can
appear in the test window by construction, and no pilot customer happens to
reappear there. The exclusion is still applied, and the exclusion file's SHA-256
is recorded in the manifest so the claim is checkable. `pilot_exclusions.json`
lives in git-ignored `outputs/`, so all 120 ids are copied into
`golden_manifest.json` — the build stays reproducible from tracked files alone.

**Near-duplicate rule (new).** D6 and D18 require near-duplicate removal; the
pilot only removed exact normalised duplicates. Implemented here as word-3-gram
Jaccard ≥ 0.80 over the D10-normalised text, greedy first-wins, with an inverted
index over shingles so only texts sharing a 3-gram are compared. No new
dependency: scikit-learn is not installed on this machine, and a TF-IDF cosine
matrix would have cost ~370 MB on a 4 GB box for a job that plain set arithmetic
does in seconds.

**Near-duplicates of the *train* window are deliberately NOT removed.** Dropping
them would delete exactly the common, templated phrasings — the easiest and most
frequent customer messages — and bias the evaluation set toward unusual text
while flattering retrieval. Train overlap is a *retrieval-time* control, not a
sampling one. What is recorded instead: **7 of the 200 customers also appear in
the train window** (`customer_also_in_train` per item in the key file). Retrieval
must exclude same-customer and same-thread evidence for those items; that is a
requirement on the retrieval stage, written down here so it cannot be forgotten.

## 3. Composition: 160 core + 40 targeted

**Core is drawn first.** In the pilot, the biased strata were drawn before
`core`, which left `core` a random sample of *the pool minus 40
non-randomly-removed items*. Here core is drawn first, so it is an honest simple
random sample of the 6,495-item pool (p = 160/6495 = 0.0246); the targeted
strata, which are intentionally biased anyway, draw from what remains. This is
the one place the golden sampler improves on the pilot's design rather than
copying it.

| Stratum | n | Eligible at draw | p(include) | Purpose |
|---|---:|---:|---:|---|
| `core` | 160 | 6,495 | 0.0246 | **Unbiased.** The only stratum a population estimate may rest on. |
| `escalation:legal_rights` | 1 | 4 | 0.250 | MUST proxy `legal_rights` |
| `escalation:safety_abuse` | 1 | 4 | 0.250 | MUST proxy `safety` |
| `boundary:account_access\|app_device_technical` | 2 | 23 | 0.087 | BR-11 |
| `boundary:content_availability\|playback_playlist` | 2 | 43 | 0.047 | BR-3 |
| `boundary:playback_playlist\|app_device_technical` | 2 | 77 | 0.026 | BR-4 |
| `boundary:product_feature_feedback\|app_device_technical` | 2 | 6 | 0.333 | BR-6 |
| `boundary:billing_subscription\|plans_eligibility` | 2 | 68 | 0.029 | BR-2 |
| `escalation:security` | 2 | 111 | 0.018 | MUST proxy `security_privacy` |
| `escalation:existing_case` | 2 | 53 | 0.038 | DM-follow-up markers |
| `escalation:failed_self_service` | 2 | 184 | 0.011 | "tried everything", "no response", reinstall/restart |
| `escalation:account_access` | 2 | 367 | 0.005 | MUST proxy `account_access` |
| `escalation:payment_dispute` | 2 | 107 | 0.019 | MUST proxy `payment_dispute` |
| `vague_unclear` | 4 | 522 | 0.008 | ≤6 normalised words — BR-8 / BR-10 territory |
| `targeted:<intent>` × 7 | 14 | 116–568 | 0.004–0.017 | one clean single-family example pair per seeded intent |
| **Total** | **200** | | | |

Boundary pairs are the confusions the 40-item pilot actually exposed
(`docs/taxonomy.md` §9.2–§9.4), not a guess: BR-11 was *added* because of them
and BR-4 was amended. A boundary item is one whose text matches **both** seed
families of the pair.

`other_unclear` has no seed family, so its targeted coverage is the
`vague_unclear` stratum — which is what BR-10 items actually look like in this
data — plus whatever `core` contributes. The pilot put 7.5% in `other_unclear`,
so core alone should contribute roughly a dozen more.

**Inclusion probability and design weight are recorded per item** (D6). Only
`core` supports a reweighted population estimate; the enriched strata have
inclusion probabilities up to 68× the core rate and must be reported separately.

### Escalation headroom

MUST-proxy hits across the finished set: `account_access` 15, `security_privacy`
6, `payment_dispute` 4, `legal_rights` 1, `safety` 1 — **22 of 200 distinct
items**, of which **9 fall in the 160-item core**. The D9 policy escalates well
beyond the MUST families (`existing_case`, `failed_self_service`,
`high_frustration`, `ambiguous`, `out_of_scope`), and the pilot's ESCALATE rate
was 35%, so the total escalation-positive count should be far higher than 22 —
but that number is a *prediction*, not a measurement, and the annotation decides
it. See §8 for what this does not support.

## 4. Strata are not labels

A stratum is a retrieval device built from observable text properties. An item
drawn as `targeted:billing_subscription` is **not** expected to be labelled
`billing_subscription`, and an item drawn as `escalation:security` is **not**
expected to be `ESCALATE`. Disagreement is a finding the validator reports, not
an error. Two of the drawn items already illustrate this:
`escalation:safety_abuse` matched on the word "threat" in a tweet about an
antivirus warning, and one `targeted:content_availability` item is really about
accidentally removing saved music.

No label, prediction, retrieval result or escalation decision was generated at
any point in this stage. No API was called.

## 5. Annotation blindness

Absent from `golden_items.csv` and `golden_items.xlsx`:

- the historical brand reply (D21 — seeing it collapses D9's separation of
  policy from observed behaviour);
- the sampling stratum;
- any intent, escalation, urgency or frustration signal;
- any retrieval result;
- thread id, customer id, tweet id, timestamp, proxy hits.

The annotation artefacts carry **`id` and `text` and nothing else**. Everything
above is in `golden_key.json` and `golden_manifest.json`, joined back only at
validation. Verification check 9 asserts the CSV's columns are exactly the locked
schema and that no cell contains a stratum name.

The 200 rows are shuffled with the build seed, so stratum membership is not
guessable from position.

## 6. Text integrity

`text` is the **exact original tweet text from `twcs.csv`**, byte-for-byte: not
normalised, not translated, not repaired, not paraphrased, not whitespace-collapsed.
Verified in check 7 by re-reading `twcs.csv` with pandas' NA coercion disabled
(so a tweet reading `nan` or `N/A` cannot become a float) and comparing all 200
strings — 200/200 identical.

This differs from the pilot, whose CSV showed the D10-normalised text. D10 fixes
normalisation for *retrieval and measurement* and states explicitly that the raw
text is what the generator sees; the annotator should judge the same string the
system will. It also matters for the labels themselves: the `public_pii`
escalation reason and the `existing_case_followup` state are partly carried by
mentions and URLs that normalisation deletes.

**Encoding.** The CSV is written programmatically as plain UTF-8 with **no BOM**
(check 10b). That is the requested format, and it is also the format Excel and
WPS read as ANSI — which is precisely how the pilot corrupted non-ASCII text in
10 of 40 rows (`docs/taxonomy.md` §9.4). 56 of these 200 rows contain non-ASCII
characters, so the same round-trip would corrupt roughly a quarter of the set.

The fix is `golden_items.xlsx`, generated programmatically from the CSV: xlsx is
UTF-8 by specification and cannot mojibake. It carries dropdown validation for
every enumerated column, wrapped text, a frozen header, and every cell forced to
text so that a tweet beginning `@`, `=`, `+` or `-` is not read as a formula.
Check 10c re-reads the xlsx and asserts all 200 texts still match the CSV.

**Annotate the xlsx.** If you prefer the CSV, use a UTF-8-aware editor and never
"Save as CSV" from a spreadsheet. `--validate` reads either UTF-8 or UTF-8-with-BOM,
and re-running `--verify` after annotation will catch any text corruption.

## 7. Deviations from pilot practice, in one place

1. **Split is test, not dev**, and not overridable (§1).
2. **Core drawn first**, so it is a true simple random sample (§3).
3. **Exact original text**, not D10-normalised text (§6).
4. **Near-duplicate removal implemented** (§2).
5. **xlsx annotation surface added**, driven by the pilot's mojibake failure (§6).
6. **D6's separate "stress" set is folded into the 40**, not held outside the
   200. D6 sketched core ~160 + MUST-enriched ~40 + a *separate* stress set;
   the boundary, vague and non-MUST escalation strata here are that stress set,
   living inside the 40-item enriched block so the total stays at 200. The
   consequence is unchanged and already stated: the enriched 40 never enter a
   headline population number, and are reported separately.
7. **`scripts/annotate_pilot.py` gained one optional argument**,
   `validate(path, keyfile=None)`, so the golden set reuses that validator
   instead of forking it. Default behaviour is unchanged; the pilot validator
   still reports 40/40, 0 errors.

Nothing in `outputs/pilot/` was modified. It is read for the exclusion list and
the 40 pilot texts (near-duplicate reference) and written to never.

## 8. What this set does and does not support

**It is not a representative sample of Spotify support traffic.** It is a
deliberately stratified evaluation set built to measure intent classification,
reply drafting and escalation quality, with targeted coverage of the boundaries
the pilot exposed. Three separate reasons it is not representative:

- 40 of 200 items are drawn from enriched strata with inclusion probabilities up
  to 68× the core rate;
- the pool is restricted to threads that **received a brand reply**;
- one item per thread and one item per customer removes 261 repeat-customer
  openers, which are a real part of the traffic.

Population statements must use the 160 core items and their design weights, and
say so.

**Known limitations to resolve before or during annotation:**

- **`legal_rights` and `safety_abuse` are structurally near-absent.** Only 4
  eligible items each in a 6,495-item pool (0.06%), and the one safety item drawn
  is a proxy false positive (an antivirus warning containing the word "threat").
  Per-reason metrics for these two codes will not be computable at any sample
  size this assignment allows. Report them as empty cells, not as zero error.
- **Escalation power rests on ~9 MUST-proxy items in the unbiased core.** This is
  D4's finding restated on real numbers: a core-only escalation rate carries a
  very wide Wilson interval. The enriched strata make the escalation *evaluation*
  meaningful; they cannot make the escalation *base rate* precise.
- **`escalation:*` and `targeted:*` proxies are crude by design** and will
  disagree with the labels. That disagreement is a reported diagnostic.
- **One annotator, one pass.** No second rater, no repeat pass, therefore no
  inter- or intra-annotator agreement. The pilot had the same gap
  (`docs/taxonomy.md` §9.4) and it is the single largest threat to the credibility
  of every number computed from this set. A re-labelled subsample of 30–40 items,
  ideally after a delay, would at least give an intra-annotator agreement figure.
- **Unresolved codebook ambiguity carried in:** P014-type
  `playback_playlist` vs `content_availability` cases with no catalogue signal,
  and P006-type entitlement-vs-client-defect cases (`docs/taxonomy.md` §9.4, §10).
  Guidance is unchanged: label the dominant reading, record the alternative in
  `notes`, and never invent a rule mid-annotation.
- **The creator/artist-distribution gap** (`docs/taxonomy.md` §6) is still
  `other_unclear` + `cannot_represent: y`. If the golden set surfaces several,
  that is a post-annotation taxonomy question, not a mid-annotation one.

## 9. How to annotate

Schema, allowed values, boundary rules and the D9 escalation policy are
unchanged from `docs/pilot.md` §4 — that section is the annotation manual and is
not restated here. In brief:

```
id,text,intent,conversation_state,urgency,frustration,language,
escalation,escalation_reason,confidence,notes,cannot_represent
```

Work top to bottom, do not look ahead, do not revise earlier rows when your
reading drifts — record the drift in `notes`. Use `confidence: low` freely;
hesitation is the signal that boundary rules are underspecified. `notes` is the
most valuable column and the pilot produced none, which is why its revisions had
to rest on label patterns instead.

Then:

```bash
python scripts/sample_golden.py --verify                                  # text integrity again
python scripts/sample_golden.py --validate data/golden/golden_items.csv   # schema + diagnostics
```

`--validate` reuses the pilot validator, so it reports the same diagnostics on
the same terms: intent distribution and thin classes, the `other_unclear` rate
against the 15% falsification threshold, low-confidence items with their notes,
`cannot_represent` items, targeted-stratum disagreements, boundary outcomes,
escalation rate and reasons, escalation-by-intent `MIXED` markers, the
account-family coherence flag, and attribute distributions.

**Do not adjust labels to make any system look good.** Nothing is being scored
here, and no system exists yet.
