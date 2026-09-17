# D12 — delayed blind re-label (test–retest)

**Status: drawn and verified 9/9. The second pass is NOT due, and was NOT run.**
Built 2026-09-14 UTC, **earliest start 2026-09-19 UTC** (5-day delay) — two days
after the 2026-09-17 submission deadline. The gate was deliberately not moved;
§5 explains why, and what it costs.

```bash
python scripts/retest_sample.py --build     # already run; rebuilding would reshuffle the draw
python scripts/retest_sample.py --verify    # integrity, blindness, delay gate
python scripts/retest_sample.py --score data/golden/retest_annotated.csv   # after the second pass
```

| File | Role | Annotator may open |
|---|---|---|
| `data/golden/retest_items.xlsx` | **30 items to re-label** — annotate here | yes, on/after 2026-09-19 |
| `data/golden/retest_items.csv` | same 30, canonical UTF-8 no BOM | yes |
| `data/golden/retest_key.json` | R id → golden id, retest stratum | **no** |
| `data/golden/retest_manifest.json` | seed, dates, quotas, source hashes | **no** |

The 200 golden annotations are untouched. The manifest records the SHA-256 of
`golden_annotated.csv` and `golden_key.json` at draw time, and `--verify` check 8
re-computes it, so any later change to the first pass is detectable.

---

## 1. Why (D12, unchanged)

One annotator, one pass, so there is no human–human reliability baseline. Without
a self-agreement figure, "the judge agrees with the human at κ = 0.62" is
uninterpretable — a weak judge and a noisy task produce the same number.
Self-agreement is the **ceiling** that makes judge–human agreement readable, and
deliverable 3 requires judge–human agreement evidence.

This set has two reliability questions of its own that a retest measures
directly: an `other_unclear` rate of 15.0% sitting exactly on the falsification
threshold, and 15 low-confidence items all sitting inside that one class
(`docs/golden_results.md` §4, §8).

## 2. How the 30 were selected

Seed `RETEST_SEED = 1212`, NumPy `default_rng`, deterministic. Drawn from the 200
golden items without replacement, **`retest:random` first** — the same
unbiased-stratum-first rule the golden sampler uses for its core, so the random
portion is a true simple random sample of all 200 rather than a sample of
whatever the enriched strata left behind.

| Stratum | n | Eligible at draw | p(include) | Why |
|---|---:|---:|---:|---|
| `retest:random` | 12 | 200 | 0.060 | **Unbiased.** The only stratum that estimates self-agreement over the set as a whole. |
| `retest:low_confidence` | 7 | 14 | 0.500 | First-pass `confidence: low` — where the annotator already said they were unsure. |
| `retest:other_unclear_confident` | 5 | 15 | 0.333 | `other_unclear` labelled at medium/high confidence — tests whether the class is a genuine judgement or a residue. |
| `retest:boundary` | 3 | 10 | 0.300 | Boundary-pair items (BR-2/3/4/6/11) — the codebook's designed hard cases. |
| `retest:escalation_deviant` | 3 | 12 | 0.250 | Items whose escalation differs from their intent's majority — the 13 rows the intent→escalation lookup gets wrong. |
| **Total** | **30** | | | |

Resulting composition of the 30 (first-pass labels, for the record — the
annotator does not see this): `other_unclear` 14, `app_device_technical` 5,
`account_access` 4, `playback_playlist` 2, `plans_eligibility` 2,
`product_feature_feedback` 2, `billing_subscription` 1; confidence
high 12 / medium 10 / low 8; ESCALATE 19 / AUTO_OK 11.

`other_unclear` is 47% of the retest against 15% of the golden set. That
enrichment is the point — and it is also why the pooled figure must not be quoted
as the headline (§4).

## 3. How blindness is preserved

- **Re-identified `R001`–`R030`.** The golden id never appears in any artefact the
  annotator opens, so the first pass cannot be looked up by id.
- **Reshuffled**, so position carries nothing.
- **Every annotation field blank**; no first-pass label is written anywhere the
  annotator can see.
- **Retest stratum withheld**, exactly as the sampling stratum was — knowing an
  item was drawn as "low confidence" would prime the second judgement.
- The R → G mapping exists only in `retest_key.json`.

`--verify` check 7 asserts that no golden id and no first-pass value appears in
the annotation CSV, and check 5 that all 30 texts are byte-identical to the
golden annotation.

**The honest limitation: recognition is still possible.** It is the same text, and
no test–retest design can prevent an annotator remembering an item. The delay is
the mitigation, not a cure. Any self-agreement figure from this procedure is
therefore an *optimistic* estimate of true label stability — memory inflates
agreement — while the hard-case enrichment pushes the pooled figure *down*. The
two biases run in opposite directions and do not cancel in any quantified way;
both are stated whenever the number is reported.

## 4. How to run it, and how to read the result

**On or after 2026-09-19**, annotate `retest_items.xlsx` under the same codebook
and the same rules as the first pass (`docs/pilot.md` §4). Do not consult the
golden set, the key, or `docs/golden_results.md` while labelling. Then export to
`data/golden/retest_annotated.csv` and run `--score`.

`--score` reports, per field: raw agreement, **Cohen's κ**, and a Wilson 95%
interval; then the `retest:random` subset alone; then per-stratum agreement; then
every disagreement with both readings side by side.

Reading rules, fixed in advance so the result cannot be framed after the fact:

- **The population ceiling comes from `retest:random`** (n=12), quoted with its
  interval. Twelve items is a wide interval and it will be reported as such.
- **The pooled 30-item figure is a conservative lower bound**, because 18 of 30
  are deliberately hard. It is not the self-agreement of the golden set.
- **Per-stratum agreement is the diagnostic.** If `retest:low_confidence` and
  `retest:other_unclear_confident` disagree heavily while `retest:random` and
  `retest:boundary` hold, the instability is localised in the unclear class —
  which is directly relevant to the §4 threshold decision still open in
  `docs/golden_results.md`.
- **Disagreements are measurements, not errors to reconcile.** Neither pass is
  edited. The golden set's first pass stays authoritative for every downstream
  metric; the second pass exists only to quantify how stable it is.

## 5. Deadline contingency — the gate was not moved

The submission deadline is **2026-09-17**. The pre-registered earliest start for
the second pass is **2026-09-19**. The two collide, and the gate was deliberately
left where it was.

The delay is not administrative padding; it is the entire mitigation for the
limitation stated in §3. Recognition of a previously-seen tweet is the one threat
test–retest cannot design away, and shortening the delay to fit a deadline would
inflate the self-agreement figure by exactly the amount the delay exists to
suppress — producing a *better-looking* ceiling that measures memory rather than
label stability. A reliability ceiling obtained that way would be worse than no
ceiling, because it would be quoted.

The earliest-start date was recorded on **2026-09-14**, in
`retest_manifest.json` (`earliest_start_utc`), before the collision was relevant.
It has not been edited since; `--verify` check 8 re-computes the SHA-256 of the
first-pass annotations recorded at draw time, so any retrospective tampering with
the source is detectable, and the date itself sits in a manifest whose hashes are
covered by the same check.

**The cost, stated plainly:** no intra-annotator reliability ceiling exists at
submission, so no agreement figure in this project — judge–human included — has
the denominator that would make it interpretable. That is a real gap in the
submission and it is reported as one in `docs/report.md` §8, not presented as a
design choice that cost nothing.

### 5.1 Verification evidence at submission

`python scripts/retest_sample.py --verify`, run 2026-09-16:

```
=== D12 retest verification ===
  [PASS] 1. row count  - 30
  [PASS] 2. R ids unique and match the key
  [PASS] 3. each R id maps to a distinct golden item  - 30 golden items
  [PASS] 4. every mapped golden id exists in the 200
  [PASS] 5. text byte-identical to the golden annotation  - 30/30
  [PASS] 6. every annotation field is blank
  [PASS] 7. blindness: no golden id and no first-pass label in the annotation CSV
  [PASS] 8. the 200 annotations are unchanged since the retest was drawn
  [PASS] 9. xlsx companion round-trips the text exactly
  [WAIT] 10. delay gate - built 2026-09-14, earliest start 2026-09-19 (3 day(s) to go)
        Not a failure: the artefact is ready, the second pass is not due yet.

=== retest strata ===
  retest:boundary                      3
  retest:escalation_deviant            3
  retest:low_confidence                7
  retest:other_unclear_confident       5
  retest:random                       12

ALL CHECKS PASSED
```

Nine integrity and blindness checks pass. The tenth is the gate, and `[WAIT]` is
the evidence that the procedure was pre-registered rather than abandoned.

## 6. What this does not do

- It does not give **inter**-annotator agreement. One person cannot produce that,
  and self-agreement is strictly the easier number — it bounds reliability from
  above, it does not establish it.
- It does not license changing the 200 labels, the taxonomy, the boundary rules,
  or the sampling design.
- With n=30 it cannot resolve per-class reliability for anything except
  `other_unclear`; the other seven intents have 1–5 items each here.
