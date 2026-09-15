# Dataset Profile — official `twcs/twcs.csv`

**STATUS: MEASURED.** Every number in the "Measured" sections below was computed
directly from the official `twcs/twcs.csv` (516,508,641 bytes, ~493 MB) on
2026-09-12. Reproduce with:

```bash
python scripts/profile.py --csv twcs/twcs.csv --leakage
```

Sections marked **Interpretation** are judgement, not measurement. They are
separated deliberately so a reviewer can disagree with the reasoning without
doubting the numbers.

Every table below is emitted by that one command; the JSON key backing each is
named in its section. Nothing here comes from a script that is not in this repo.

---

## 1. Verification against `docs/research.md`

The prior profiling extracts in `docs/research.md` carried a provenance warning:
they were obtained through a side channel because Kaggle and Hugging Face were
unreachable from the analysis environment. **That warning can now be retired.**
Re-derived from the official CSV, the prior numbers replicate:

| Statistic | `research.md` | Measured | Status |
|---|---|---|---|
| SpotifyCares brand tweets | 43,265 | 43,265 | exact |
| SpotifyCares customer tweets | 48,543 | 48,543 | exact |
| SpotifyCares threads | 28,280 | 28,280 | exact |
| SpotifyCares openers | 28,221 | 28,221 | exact |
| Threads of exactly 2 tweets | 62.9% | 62.9% | exact |
| Thread size median / p90 | 2 / 6 | 2 / 6 | exact |
| Exact duplicate brand replies | 24.5% | 24.5% | exact |
| Brand replies containing signatures | 97.1% | 97.1% | exact |
| Brand replies containing links | 50.5% | 50.5% | exact |
| Openers containing links | 15.7% | 15.7% | exact |
| First action private/DM | ~37.2% | 37.0% | agrees |
| **Test reply verbatim in train** | **21.4%** | **21.4%** | exact |
| AmericanAir brand tweets | 36,764 | 36,764 | exact |
| AmericanAir customer tweets | 50,054 | 50,054 | exact |
| AmericanAir threads | 26,386 | 26,386 | exact |
| AmericanAir first action private | 17.4% | 16.3% | agrees |

Three figures differ, all by **definition rather than error**:

- **Customer follow-up rate.** `research.md` 41.9% (Spotify) / 47.8% (AA);
  measured 35.1% / 42.8%. Here a follow-up means a thread containing more than
  one customer tweet. Definition now fixed in `scripts/profile.py`.
- **Repeat customers.** `research.md` 12.7%; measured 7.6%. Measured as
  `1 − distinct opener authors / openers`.
- **Threads straddling the test boundary.** `research.md` 288; measured **0**.
  Not a discrepancy to resolve — a design change. Splitting on *opener* date and
  assigning the whole thread to its opener's window makes straddling impossible
  by construction. This is the safer split and is now the documented approach.

The prior model probes (TF-IDF DM-proxy AUC ~0.889 etc.) have **not** been
re-run and remain unverified. They are not used in any decision below.

---

## 2. Measured — dataset level

| Property | Value |
|---|---|
| Rows | 2,811,774 |
| Distinct authors | 702,777 |
| Inbound (customer) share | 54.7% |
| Rows with `in_response_to_tweet_id` | 71.7% |
| Reply edges resolved | 2,013,577 / 2,017,439 = **99.81%** (3,862 dangling) |
| Distinct conversation threads (whole corpus) | 798,197 |
| Distinct outbound brand accounts | 108 |

Top brands by outbound volume: AmazonHelp 169,840 · AppleSupport 106,860 ·
Uber_Support 56,270 · **SpotifyCares 43,265** · Delta 42,253 · Tesco 38,573 ·
**AmericanAir 36,764** · TMobileHelp 34,317 · comcastcares 33,031 ·
British_Airways 29,361.

The 3,862 dangling parent references are tweets replying to something outside
the dataset. At 0.19% they are immaterial; they become isolated thread roots.

---

## 3. Measured — SpotifyCares vs AmericanAir

### Structure

| | SpotifyCares | AmericanAir |
|---|---|---|
| Threads | 28,280 | 26,386 |
| Brand tweets | 43,265 | 36,764 |
| Customer tweets | 48,543 | 50,054 |
| Tweets in those threads | 91,889 | 87,584 |
| Thread size median / mean / p90 | 2 / 3.25 / 6 | 2 / 3.32 / 6 |
| Threads of exactly 2 | 62.9% | 56.1% |
| Root is a customer | 99.8% | 99.1% |
| Customer follow-up rate | 35.1% | 42.8% |
| Openers | 28,221 | 26,148 |
| Date span | 2013-09-18 → 2017-12-03 | 2011-06-16 → 2017-12-03 |
| Share within Oct 1 – Dec 3 2017 | 99.5% | 99.7% |

### Brand reply character — the drafting task

| | SpotifyCares | AmericanAir |
|---|---|---|
| Duplicate replies (URL/mention masked) | 17.8% | 3.2% |
| **Duplicate replies (signature stripped)** | **24.5%** | **3.2%** |
| Contains an agent signature | **97.1%** | **0.0%** |
| Contains a link | 50.5% | 6.5% |
| DM / private deflection (all replies) | 30.8% | 16.8% |
| Median reply length (chars) | 131 | 107 |
| First action is DM/private | 37.0% | 16.3% |
| First action contains a link | 54.3% | 6.9% |
| First-reply latency median / p90 (min) | 65.1 / 262.2 | 12.3 / 49.2 |

Most common Spotify reply templates (signature and URL stripped, per D10) — the
top 10 templates together cover only 2.56% of replies, across 32,651 distinct
normalised templates, so duplication is **diffuse**: many small repeats rather
than a handful of mega-templates. Emitted as `brands.*.reply_templates`:

```
183x  hi there! can you dm us your account's email address or username? we'll take a look
142x  hey! fingers crossed we'll be able to have it soon, but there's info about spotify conte...
120x  hey there! can you dm us your account's email address? we'll take a look
```

### Openers — the classification task

| | SpotifyCares | AmericanAir |
|---|---|---|
| Median opener length (chars) | 111 | 130 |
| Opener contains a link | 15.7% | 20.6% |
| Repeat customers among openers | 7.6% | 18.5% |
| Openers receiving a brand reply | 100% | 100% |

Keyword prevalence over **all** openers (proxy families, case-insensitive;
28,221 Spotify / 26,148 AmericanAir openers, whole date range — not the test
window). Emitted as `brands.*.opener_topics_all`:

| Family | SpotifyCares | AmericanAir |
|---|---|---|
| money / refund | 20.1% | 4.7% |
| account / login | 18.1% | 0.5% |
| technical / bug | 5.0% | 2.8% |
| anger / profanity | 3.5% | 7.3% |
| praise | 5.9% | 14.5% |
| delay / cancel | 2.4% | 11.9% |
| baggage | 0.0% | 9.8% |

---

## 4. Measured — leakage, duplication and retrieval

Temporal split on **opener date**: train 2017-10-01→11-05, dev 11-06→11-19,
test 11-20→12-03. Openers retained only where a brand reply exists.

| | SpotifyCares | AmericanAir |
|---|---|---|
| Split sizes (train / dev / test) | 14,547 / 6,716 / 6,808 | 14,764 / 5,622 / 5,680 |
| Test customers also seen in train | **2.87%** | 8.59% |
| Test threads straddling the boundary | 0 | 0 |
| **Test gold reply appears verbatim in train** | **21.4%** | 1.9% |

TF-IDF 1-NN retrieval probe over openers (train → test):

| | SpotifyCares | AmericanAir |
|---|---|---|
| Opener similarity, median | 0.303 | 0.255 |
| Opener similarity ≥ 0.9 | 1.4% | 0.7% |
| **Retrieved reply == gold reply** | **0.4%** | 0.0% |
| **Retrieved-vs-gold reply cosine, median** | **0.040** | 0.018 |

**Interpretation.** These two rows must be read together, and the combination is
the single most decision-relevant finding in this profile.

21.4% of Spotify's test gold replies exist verbatim somewhere in train, so the
*reply distribution* is heavily templated. But a simple retriever recovers the
gold reply only 0.4% of the time, and the median similarity between what it
retrieves and the gold reply is 0.040 — effectively unrelated text.

Two consequences follow. First, the 21.4% is **not** an exploitable retrieval
leak through opener similarity; it is a property of the reply distribution. The
real exposure is a system that scores well by emitting whichever template is
most common, which must be checked directly rather than assumed away. Second,
and more important: the mapping from opener to reply is **high-entropy**. Many
different replies are legitimate for the same opener, and the single historical
reply is one sample from a wide distribution. Any metric scoring a draft by
overlap with that one reply is close to uninformative, and would punish good
drafts for not guessing which template an agent happened to use.

Caveat: TF-IDF 1-NN is a weak retriever. These numbers set an honest floor, not
a ceiling. A stronger retriever may raise them, which is precisely why any
claimed retrieval gain has to be demonstrated rather than assumed.

---

## 5. Measured — escalation prevalence and statistical power

Keyword **proxy** for MUST-escalation content, over test-window openers. These
are proxies used to size the labelling problem. They are never used as labels,
training targets or evaluation ground truth.

| Family | SpotifyCares | AmericanAir |
|---|---|---|
| account_access | 5.6% | 0.1% |
| payment_dispute | 1.6% | 1.4% |
| security_privacy | 1.7% | 0.3% |
| legal_rights | 0.1% | 0.1% |
| safety | 0.1% | 0.6% |
| **any MUST proxy** | **7.5%** | **2.5%** |

Resulting power for a MUST-recall headline, assuming a randomly sampled core
golden set and true recall of 90% (Wilson 95% interval):

| Core size | Brand | n_MUST | 95% CI | half-width |
|---|---|---|---|---|
| 160 | SpotifyCares | ~12 | [0.65, 0.99] | ±17.0pp |
| 200 | SpotifyCares | ~15 | [0.70, 0.99] | ±14.3pp |
| 160 | AmericanAir | ~4 | [0.51, 1.00] | ±24.5pp |
| 200 | AmericanAir | ~5 | [0.38, 0.96] | ±29.4pp |

Paired McNemar sizing, at 80% power and an assumed 20% discordance rate,
for detecting a recall gap against a baseline:

| True gap | n_MUST required |
|---|---|
| 15pp | ~57 |
| 10pp | ~145 |
| 5pp | ~616 |

**Interpretation.** MUST-recall cannot be the headline metric under random core
sampling for either brand. At Spotify's 7.5% proxy prevalence a 200-item core
yields ~15 MUST items; at AmericanAir's 2.5% it yields ~5. The assignment caps
the golden set at 250 items, and detecting even a 10pp recall difference would
need ~145 MUST items. The headline must move to a metric whose denominator is
large, and MUST items must be deliberately over-sampled rather than left to
chance. Both changes are recorded in `docs/decisions.md`.

The proxy under-counts. Human policy labelling will mark items as escalate-worthy
that these narrow regexes miss — Spotify's 24.2% billing mass (below) plainly
contains payment disputes my `payment_dispute` pattern does not match. True
prevalence is therefore somewhere above 7.5%, plausibly 15–30%, but unknown
until the pilot. The direction of the conclusion does not change.

### Candidate intent prevalence (test-window openers)

A class is treated as viable at ≥4% prevalence (≈8 examples in a 200-item core).

| Candidate | SpotifyCares | AmericanAir |
|---|---|---|
| billing_subscription | **24.2%** | 1.1% |
| content_availability | **11.4%** | 0.9% |
| praise_social | **8.5%** | **16.8%** |
| playback_technical | 3.9% | 0.8% |
| seat_booking | 0.7% | **10.2%** |
| baggage | 0.0% | **8.6%** |
| delay_cancel | 1.9% | **5.1%** |
| loyalty | 0.5% | **4.1%** |
| Viable classes (≥4%) | 3 | 5 |

---

## 6. Measured — temporal drift

Share of first brand replies that deflect to DM, by split:

| Split | SpotifyCares | AmericanAir |
|---|---|---|
| train | 0.369 | 0.166 |
| dev | 0.313 | 0.152 |
| test | **0.431** | 0.163 |

**Interpretation.** Spotify's historical DM behaviour is not stationary: it dips
in the dev window and then rises ~6.2pp above train in the test window. Any
model fitted on train behaviour faces a shifted test distribution, and any
metric defined against historical DM behaviour will move for reasons that have
nothing to do with system quality. This is recorded as a named evaluation risk.
AmericanAir is stable by comparison (0.166 / 0.152 / 0.163).

This also reinforces a standing decision: historical DM behaviour is *behaviour*,
not escalation ground truth, and is reported separately under that name.

---

## 7. Interpretation — brand recommendation

**Recommended: SpotifyCares.** AmericanAir wins most individual columns, so the
reasoning matters more than the verdict.

**Where AmericanAir is genuinely better.** It is the better drafting corpus by a
wide margin: 3.2% duplicate replies against 24.5%, no agent signatures at all
against 97.1%, 6.5% links against 50.5%, and less DM deflection. Its replies are
substantive prose. It also supports a better intent taxonomy — 5 viable classes
against 3, spread more evenly. Its behaviour is temporally stable. Its gold
replies are essentially never reused verbatim (1.9%).

**Why SpotifyCares is still the better basis for this assignment.**

1. **Escalation has actual positive mass.** 7.5% MUST proxy against 2.5% — 2.9×
   the density, concentrated in account access (5.6% vs 0.1%) and billing
   (24.2% vs 1.1%). Building an escalation system for a brand where almost
   nothing requires escalation is not a sampling problem that can be fixed; the
   mass is not there. Escalation is one of the three required functions and is
   where the strongest proof is available.
2. **The corpus matches the intended architecture.** Spotify's 37% DM-deflection
   rate looks like a thin drafting task until you note that the design already
   routes escalated cases to *deterministic handoff templates*. Those cases never
   reach the generator. The generative task is the auto-handled remainder —
   content availability 11.4%, playback 3.9%, praise 8.5% — which is genuinely
   substantive. The data's shape supports the design rather than fighting it.
3. **Cleaner customer separation.** 2.87% test/train customer overlap against
   8.59%, so the temporal split needs less repair.
4. **Richer material for the mandatory "what is misleading about my headline
   number?" section.** Template reuse at 21.4%, a retrieval floor of 0.4%, and
   real DM-rate drift give three evidence-backed answers rather than
   hand-waving.

**The cost, stated plainly.** Spotify gives a thinner intent taxonomy (3 viable
classes by keyword proxy against 5) and a more templated drafting corpus. The
taxonomy must be derived from the data after the brand is locked, not from these
regexes; the 24.2% billing mass plausibly splits into billing versus
subscription-management, and content availability at 11.4% is healthy. If the
pilot shows fewer than ~6 workable classes, revisiting this choice is legitimate.

---

## 8. Limitations of this profile

- All keyword families are **proxies**, tuned by hand, with unmeasured recall and
  precision. They size the problem; they are not labels.
- The non-English estimate from earlier exploration (~4.5% in `research.md`) is
  **not** reproduced here. A character-range heuristic returned 0.2% / 0.1%,
  which disagrees enough that neither figure should be trusted. Needs a proper
  language detector before any filtering decision.
- Thread reconstruction uses `in_response_to_tweet_id` only. Threads linked only
  by `response_tweet_id` in the opposite direction are not cross-checked.
- Signature detection is a regex over sign-off shapes. At 97.1% for Spotify and
  0.0% for AmericanAir the contrast is real, but the exact rate carries some
  pattern-dependent error.
- The split boundaries (Oct 1 / Nov 6 / Nov 20 / Dec 4) are inherited from prior
  work and are not themselves optimised. ~99.5% of both brands' data falls in a
  nine-week window, so there is little room for a different sensible split.
- No model has been trained and no API has been called. Nothing here is a
  performance result.
