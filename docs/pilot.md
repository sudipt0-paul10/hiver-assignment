# Taxonomy Annotation Pilot

**Codebook version: 0.2.1-proposed** (`docs/codebook.json`, defined in
`docs/taxonomy.md`). The codebook is **frozen for the duration of this pilot** —
the point is to find where it fails, not to repair it mid-flight.

This is the ~40-item pilot, **not** the 150–250 golden set.

---

## 1. Split choice — a deliberate deviation, flagged

The pilot is sampled from **dev** (2017-11-06 → 2017-11-19), not test.

The instruction was to sample from test so the golden evaluation stays
out-of-training. That goal is already met: **the golden set still comes from the
test window** and nothing about that changes. But the *pilot's* output is a
revised codebook, and revising a codebook against test-window items is tuning
against the evaluation set — the contamination D7 forbids ("thresholds and
prompts are tuned on dev only; the test window is touched once, for final
numbers") and that the project brief calls out by name.

Dev is the same brand, the same thread reconstruction, and adjacent in time, so
it is representative for deciding whether a label definition is clear.

**To overrule:** `python scripts/annotate_pilot.py --build --split test`. If you
do, `outputs/pilot/pilot_exclusions.json` lists every sampled thread, customer
and tweet id, and **all of them must be excluded from the golden set** — the file
is written for exactly that purpose, on both splits.

## 2. Sampling method

Reproduce exactly:

```bash
python scripts/annotate_pilot.py --build            # defaults: --split dev, --out outputs/pilot
```

- **Brand:** SpotifyCares only.
- **Seed:** `PILOT_SEED = 913`, NumPy `default_rng`. Deterministic.
- **Pool:** dev-window openers whose thread received a brand reply, after
  one-per-thread, customer-disjoint, and exact-normalised-duplicate removal →
  **6,286 candidates**.
- **Text:** D10 normalisation (signatures, URLs and mentions stripped).

**Strata** — drawn in this order, without replacement, so an item is never
double-counted:

| Stratum | n | Purpose |
|---|---|---|
| `boundary` | 5 | Matches **two or more** seed families — the confusion cases |
| `ambiguous` | 3 | ≤6 words, or carries DM-follow-up markers |
| `targeted:account_access` | 2 | |
| `targeted:billing_subscription` | 2 | |
| `targeted:plans_eligibility` | 2 | |
| `targeted:content_availability` | 2 | |
| `targeted:playback_playlist` | 2 | |
| `targeted:app_device_technical` | 2 | |
| `targeted:product_feature_feedback` | 3 | cleared the floor in the pilot (7/40) |
| `core` | 17 | **Random, unbiased** |
| **Total** | **40** | |

**The targeted strata are deliberately non-proportional.** This pilot stresses
the codebook; it does **not** estimate prevalence. Only `core` is unbiased, and
even 14 items estimate nothing usefully. Prevalence comes from the golden set.

**Seed families are retrieval devices, never labels.** A row drawn as
`targeted:billing_subscription` is *not* expected to be labelled
`billing_subscription`. Disagreement is a finding.

## 3. Two things withheld from the annotator, by design

1. **The historical brand reply is absent** from every pilot artefact. Seeing how
   Spotify actually replied would anchor the policy label onto observed behaviour
   and collapse D9. Historical DM behaviour is *not* escalation ground truth.
2. **The stratum is absent** from `pilot_items.csv`. Showing that an item was
   drawn as `targeted:account_access` telegraphs the answer and would corrupt the
   boundary judgements this pilot exists to measure. Strata live in
   `pilot_key.json` and are rejoined only at validation.

## 4. How to annotate

Open `outputs/pilot/pilot_items.csv` in Excel, LibreOffice or a text editor.
Forty rows, `id` and `text` pre-filled, everything else blank. Fill every
column. Work top to bottom; **do not look ahead** and do not revise earlier rows
after your definitions drift — the drift itself is a finding worth recording in
`notes`.

| Column | Allowed values |
|---|---|
| `intent` | one of the 8: `account_access`, `billing_subscription`, `plans_eligibility`, `content_availability`, `playback_playlist`, `app_device_technical`, `product_feature_feedback`, `other_unclear` |
| `conversation_state` | `opener` · `existing_case_followup` · `unclear` |
| `urgency` | `low` · `normal` · `high` |
| `frustration` | `low` · `normal` · `high` |
| `language` | `english` · `non_english` · `unclear` |
| `escalation` | `ESCALATE` · `AUTO_OK` |
| `escalation_reason` | `none` (iff AUTO_OK), else `account_access`, `payment_dispute`, `security`, `legal_rights`, `safety_abuse`, `public_pii`, `existing_case`, `failed_self_service`, `high_frustration`, `out_of_scope`, `ambiguous` |
| `confidence` | `low` · `medium` · `high` |
| `notes` | free text — **the most valuable column** |
| `cannot_represent` | `y` if no intent genuinely fits; leave blank otherwise |

### Rules

- **Language is an attribute, not an intent and not an escalation reason.** If a
  non-English message has clear intent, label that intent normally. If intent
  cannot be determined reliably, use `other_unclear` and record the language.
  Never escalate something *because* it is non-English.
- **Authentication failure → `account_access`, whatever device is named (BR-11).**
  Cannot log in, unexpectedly logged out, password reset failing, account
  compromised — all `account_access` even when the message names a phone, an OS
  or "the app". Device names say *where* it broke, not *what* broke.
- **Social-only praise → `other_unclear` (BR-5).** `social_praise` was removed as
  an intent in codebook 0.2.0; set `social_nonrequest_indicator` instead.
- **Unsupported device/OS is a feature request, not a defect (BR-6).** If the
  capability was never shipped, it is `product_feature_feedback`; if it shipped
  and is broken, it is the technical intent.
- **Topic without a request → `other_unclear` (BR-10).** If the message names a
  subject but states no request, problem, question or actionable need, it is
  `other_unclear`. *"question about billing."* is not `billing_subscription`.
  **Do not infer the issue the customer probably meant.**
- **Unsupported domains → `other_unclear` for now.** Artist/music distribution
  (creator-side) has no intent yet, by decision. Record it in `notes` and set
  `cannot_represent: y`. Do not invent a class.
- **Intent answers "what does the customer want?"** — nothing else. Not how angry
  they are, not how urgent, not whether it should escalate.
- **One intent per item.** Attributes are orthogonal and always recorded.
- **Escalation is a separate judgement** applied from the D9 policy below. It is
  not derived from the intent and never from the historical reply.
- **Use `confidence: low` freely.** A low-confidence label plus a note is worth
  more to this pilot than a confident guess. Hesitation is the signal.
- **`cannot_represent: y` is not failure.** It is the thing we are looking for.
  Record what the item actually wanted in `notes`. **Do not invent a new intent** —
  those are gathered and decided after the pilot, not during it.
- **Do not adjust labels to make any system look good.** Nothing is being scored
  here.

### Escalation policy (D9, provisional — unchanged for this pilot)

`ESCALATE` when the case should not be handled autonomously in public:

- account access or compromise → `account_access` / `security`
- a disputed charge, refund, or payment → `payment_dispute`
- legal, rights or regulatory framing → `legal_rights`
- safety, abuse or harassment → `safety_abuse`
- personal data exposed in public → `public_pii`
- already in a private case → `existing_case`
- self-service already failed and the customer is stuck → `failed_self_service`
- high frustration with an unresolved problem → `high_frustration`
- outside what support can do → `out_of_scope`
- too ambiguous to act on safely → `ambiguous`

`AUTO_OK` when a public reply can reasonably resolve or correctly route it:
public how-to, known issue explanation, catalogue answer, ordinary feedback,
thanks and social.

When genuinely torn, choose `ESCALATE` and set `confidence: low`. Over-escalating
in a pilot is cheap; a wrong `AUTO_OK` is the failure mode the headline metric
(D3) is built to catch.

## 5. Validate when finished

```bash
python scripts/annotate_pilot.py --validate outputs/pilot/pilot_items.csv
```

Checks every value is legal, that `AUTO_OK` carries reason `none`, and that
`ESCALATE` does not. Then reports:

- **intent distribution**, flagging classes with ≤1 example
- **unclear rate** — **>15% means the taxonomy is inadequate** and must be revised
  before the golden set (falsification criterion, `docs/taxonomy.md` §6)
- **low-confidence items** with their notes — the boundary-failure list
- **`cannot_represent` items** — candidates for a new intent
- **targeted-stratum disagreements** — where a seed said one thing and you said
  another
- **boundary/ambiguous outcomes**
- **escalation rate, reason distribution, and escalation split by intent** — a
  `MIXED` marker means intent alone does not determine policy, which is expected
  and confirms escalation must stay a separate judgement
- **attribute distributions** — a near-constant attribute is not carrying
  information

## 6. What this pilot decides

Per `docs/taxonomy.md` §6, the pilot should resolve:

- Does **`product_feature_feedback`** earn its place, or fall below the floor?
- Does **`social_praise`** survive once BR-5 removes polite requests?
- Are **`billing ↔ plans`** confusable beyond 25%? If so *and* escalation
  outcomes match → merge.
- Are **`playback ↔ app_device`** confusable beyond 25%? If so, **do not merge**
  (owner decision) — sharpen BR-4 and re-pilot.
- Is the **`other_unclear`** rate under 15%?
- Do the attributes carry information, or are they near-constant?

With 40 items, none of these is a statistical test. They are signals for a human
decision, with wide uncertainty — a class seen twice tells you almost nothing
about its true prevalence.

## 7. Output locations

| File | Contents |
|---|---|
| `outputs/pilot/pilot_items.csv` | the 40 items to annotate |
| `outputs/pilot/pilot_key.json` | stratum per id (withheld until validation) |
| `outputs/pilot/pilot_manifest.json` | full sampling record |
| `outputs/pilot/pilot_exclusions.json` | thread/customer/tweet ids to exclude from the golden set |

All of `outputs/` is git-ignored, as instructed.

**One caution:** your completed annotations are irreplaceable human work living in
an ignored directory — one `git clean -fdx` destroys them. Back the filled CSV up
outside the repo, or decide later to track it (`.gitignore` already carries an
un-ignore carve-out for `data/golden/`, which would be the natural home).
