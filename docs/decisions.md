# Architecture & Evaluation Decisions

**STATUS: REVISED 2026-09-12** against the official `twcs/twcs.csv`.

Decisions marked **[MEASURED]** rest on evidence in `docs/profile.md`, which is
reproducible via `scripts/profile.py`. Decisions marked **[POLICY]** are choices
we are making, not facts we discovered. Decisions marked **[OPEN]** are not yet
settled.

Numbered for reuse as the assignment's required decision log (10–15 non-obvious
decisions with rationale).

---

## D1 — Brand: SpotifyCares **[MEASURED]**

SpotifyCares (43,265 brand tweets, 28,280 threads, 28,221 openers).

**Why, despite AmericanAir winning most columns.** MUST-escalation proxy
prevalence is 7.5% for Spotify against 2.5% for AmericanAir — 2.9× the density,
concentrated in account access (5.6% vs 0.1%) and billing (24.2% vs 1.1%).
Escalation is one of the three required functions; a brand where almost nothing
needs escalating has no positive mass to measure, and that is not fixable by
sampling. Spotify also has cleaner customer separation (2.87% vs 8.59% test/train
overlap).

**AmericanAir trade-offs, accepted explicitly.** AmericanAir is the better
drafting corpus: 3.2% duplicate replies vs 24.5%, 0.0% signatures vs 97.1%, 6.5%
links vs 50.5%, 1.9% verbatim reply reuse vs 21.4%, and temporally stable DM
behaviour. It also supports 5 viable intent classes vs 3. We are giving up a
cleaner generation task and a richer taxonomy to gain a measurable escalation
task.

**Why the drafting cost is smaller than it looks.** Escalated items are answered
by deterministic handoff templates (D8), so Spotify's 37% DM-deflection cases
never reach the generator. The generative task is the auto-handled remainder.

**Revisit trigger.** If the pilot yields fewer than ~6 workable intent classes,
reopen this decision.

## D2 — Primary escalation label: binary `ESCALATE` / `AUTO_OK` **[POLICY]**

The primary label is binary. Rationale is twofold and both arguments point the
same way:

- **Power.** MUST alone is too rare to headline (D4). `ESCALATE` = MUST ∪ SHOULD
  has a large enough denominator to measure.
- **Label noise.** The MUST/SHOULD boundary is exactly the distinction most
  likely to show poor intra-annotator reliability, and the headline should not
  depend on our shakiest judgement. The `unsafe-auto` definition already treats
  MUST and SHOULD together.

Binary is also closer to the assignment's own wording ("auto-handling vs. human
escalation"). The three-level `MUST` / `SHOULD` / `AUTO_OK` scale and the reason
codes (D9) are retained and reported as **secondary** analysis.

## D3 — Headline metric: unsafe-auto rate at matched auto-handling coverage **[POLICY]**

**Definition, stated precisely:**

> **unsafe-auto rate** = of all items the system chooses to handle automatically,
> the percentage that human policy labels `ESCALATE`.
>
> denominator = items auto-handled by the system
> numerator = those items whose human policy label is `ESCALATE`

> **coverage** = percentage of all evaluated items the system chooses to handle
> automatically.

Unsafe-auto rate is meaningless without coverage — a system that auto-handles
nothing has an unsafe-auto rate of 0%. Systems are therefore always compared **at
matched coverage**, and the full curve is reported (D5).

Chosen over MUST-recall because its denominator is the large auto-handled set
rather than the ~15-item MUST set, and because it is the operationally
meaningful safety number: it answers "when this system acts alone, how often
should it not have?"

## D4 — MUST-recall is no longer the headline **[MEASURED]**

Superseded. At Spotify's 7.5% MUST proxy prevalence, a 200-item random core
yields ~15 MUST items: 90% recall carries a 95% Wilson interval of [0.70, 0.99],
±14.3pp. Detecting a 10pp recall gap against a baseline by paired McNemar would
need ~145 MUST items; a 5pp gap ~616. The assignment caps the golden set at 250.

MUST-recall is still **reported**, with its interval, as a secondary safety
metric. It is not the number any claim rests on.

## D5 — Report the complete risk–coverage curve **[POLICY]**

The headline is a curve, not a point. Plot unsafe-auto rate against coverage
across the decision threshold, with these reference points marked:

- **always-escalate** — coverage 0%, unsafe-auto 0% (trivially safe, useless)
- **never-escalate** — coverage 100%, unsafe-auto = the `ESCALATE` base rate
- **keyword rule** — the simple baseline, as a single point
- our system's operating point, chosen on **dev**, never on test

Reporting both endpoints makes the trade-off legible and prevents a favourable
threshold from masquerading as a win.

## D6 — Golden set: stratified, weights recorded **[POLICY]**

Approximately 200 items total, inside the assignment's 150–250 range:

| Stratum | Size | Purpose |
|---|---|---|
| **Core** | ~160 | Random sample of test-window openers. The unbiased population estimate. |
| **MUST-enriched** | ~40 | Over-sampled likely-escalation items, so escalation metrics have positive mass. |
| **Stress** | separate | Follow-ups, rare/high-risk cases (security, legal, non-English). |

Rules: one item per thread; customer-disjoint across strata; near-duplicates
removed; drawn only from the test window.

**Sampling weights are recorded per item** (inclusion probability under its
stratum), so population-level estimates are reweighted and unbiased. Without
this, over-sampling MUST silently inflates the reported escalation base rate.

The stress set is **intentionally biased** and is reported separately. It never
enters a headline number.

Reduced from the previous 240-item proposal: the marginal statistical value of
items 200→240 is negligible (D4), and that labelling effort buys far more as
judge-validation ratings (D11).

## D7 — Statistics: paired tests and intervals **[POLICY]**

- **Paired McNemar** for every system-vs-baseline comparison, on identical items.
  Comparing two independent point estimates over ~200 items would be
  underpowered and misleading.
- **Wilson 95% intervals** on every reported proportion. No bare point estimate
  appears in the report without one.
- Thresholds and prompts are tuned on **dev only**. The test window is touched
  once, for final numbers.

## D8 — Pipeline **[POLICY]**

```
customer message
  → text normalisation (D10)
  → intent classification
  → ESCALATE / AUTO_OK decision  (+ reason)
  → if AUTO_OK: retrieve similar historical cases → LLM draft → guardrails
  → if ESCALATE: deterministic handoff template (no generation)
```

Escalated cases receive a deterministic handoff rather than a generated reply
that pretends to resolve a private or account-specific issue. This is a safety
property, and it removes Spotify's DM-deflection mass from the generative task.

Retrieval starts at TF-IDF / BM25. Embeddings or a vector store are added only
if evaluation demonstrates a meaningful gain (D15).

## D9 — Escalation is our policy, not observed behaviour **[POLICY]**

TWCS contains **no escalation ground truth**. Our `ESCALATE` / `AUTO_OK` labels
are a written policy applied by a human annotator against a locked codebook.

Historical DM behaviour is **secondary evidence only** and is always reported
under the name *behavioural agreement*, never as accuracy against truth. D16
shows why this matters: Spotify's DM rate is not even stationary.

Three-level scale (secondary): `MUST` (must not be handled autonomously),
`SHOULD` (likely needs human review), `AUTO_OK` (safe for autonomous public
handling).

Reason codes: `account_access`, `payment_dispute`, `security`, `legal_rights`,
`safety_abuse`, `public_pii`, `existing_case`, `failed_self_service`,
`high_frustration`, `out_of_scope`, `ambiguous`.

No arbitrary recall target is imposed; the operating point comes from dev data
and an explicit risk statement.

## D10 — Mandatory text normalisation **[MEASURED]**

Before **retrieval and evaluation**, and only there:

1. strip agent signatures (`^JD`, `/RS`, `- Ben`)
2. mask URLs → `<URL>`
3. mask mentions → `<USER>`
4. collapse whitespace, lowercase for matching

**The raw original text is always preserved and is what the generator sees.**

Not cosmetic: 97.1% of Spotify's brand replies carry a signature and 50.5%
carry a link. Unnormalised, similarity metrics would partly measure signature
and URL matching, and the model would be penalised for failing to invent URLs it
cannot know. The measured duplicate rate itself moves with normalisation —
17.8% URL-masked vs 24.5% signature-stripped — which is exactly why the contract
is fixed in one place (`scripts/profile.py`) rather than re-improvised per script.

## D11 — Judge validation: human ratings plus corruption probes **[POLICY]**

- **~60–80 human reply-quality ratings.** A distinct labelling task from the
  golden set: the golden set labels intent and escalation, the judge rates reply
  *quality*. The previous plan budgeted nothing for this despite judge–human
  agreement being a required deliverable.
- **Deliberately corrupted-reply probes.** Inject known defects — wrong intent,
  fabricated facts, missing escalation, hostile tone. A judge that cannot detect
  planted corruptions is disqualified regardless of its correlation with humans.
  This is the sharper of the two tests.
- Rubric: addresses need · grounded in historical evidence · appropriate routing ·
  tone · overall acceptability. Frozen after **one** revision.
- Agreement reported as a coefficient with an interval, against D12's ceiling.

## D12 — ~30 test–retest items for intra-annotator reliability **[POLICY]**

With a single annotator there is no human–human reliability baseline, so
"the judge agrees with the human at κ = 0.6" is uninterpretable — it could mean a
poor judge or an inherently noisy task.

Re-label ~30 items after a deliberate delay, blind to the first pass. This
self-agreement is the **ceiling** against which judge–human agreement is read.
Cheap, and it converts an uninterpretable number into an interpretable one.

## D13 — No BLEU / ROUGE / reference-overlap in the headline **[MEASURED]**

Reference-overlap metrics are excluded from the headline on measured grounds,
not preference.

TF-IDF 1-NN retrieval over Spotify openers (train → test):

- retrieved reply equals the gold reply **0.4%** of the time
- median cosine between retrieved and gold reply **0.040**
- median opener-to-opener similarity 0.303

Meanwhile 21.4% of test gold replies appear verbatim *somewhere* in train. Read
together: the reply distribution is templated, but the opener does not predict
which template an agent used. The opener→reply mapping is high-entropy and the
single historical reply is one sample from a wide distribution of acceptable
replies. Scoring a draft by overlap with that one reply would punish good drafts
for guessing a different valid template.

This is the empirical justification for judge-based evaluation (D11). Reference
metrics may appear as a reported secondary curiosity, clearly labelled.

**Caveat recorded:** TF-IDF 1-NN is a weak retriever. 0.4% is an honest floor,
not a ceiling — which is why any retrieval gain must be demonstrated (D15).

## D14 — Baselines **[POLICY]**

The assignment requires at least two. We commit to:

- **Trivial:** always-escalate, never-escalate, majority-class, constant-DM
- **Simple:** keyword rules; TF-IDF + logistic regression; 1-NN retrieval

Every baseline is evaluated on the identical golden items and compared by paired
McNemar (D7).

## D15 — Complexity must earn its place **[POLICY]**

No embeddings, no vector database, no GPU, no fine-tuning, no UI framework, no
orchestration framework. Each is added only if an ablation on dev shows a
meaningful gain over the simpler component it replaces.

TF-IDF and logistic regression are CPU-cheap, explainable live, and — per D13 —
the retrieval ceiling is unproven anyway. The machine has 2 cores, 3 GB RAM and
no GPU, so local model inference is not a realistic option regardless.

## D16 — Recorded evaluation risk: Spotify's DM-rate drift **[MEASURED]**

Historical first-reply DM rate by split: **train 0.369 / dev 0.313 / test 0.431**.

The test window sits ~6.2pp above train and ~11.8pp above dev. Consequences:

- any model fitted to train behaviour meets a shifted test distribution
- any metric defined against historical DM behaviour moves for reasons unrelated
  to system quality
- it is direct evidence for D9: this behaviour is not a stable ground truth

Belongs in the report's "what is misleading about my headline number?" section.

## D17 — Intent taxonomy: 9 proposed classes, 5 attributes separated **[MEASURED / PROPOSED]**

Superseded the earlier open question. Full codebook in `docs/taxonomy.md`;
machine-readable form in `docs/codebook.json` (version **0.1.1-proposed**).
Derived from the **train split only** (14,547 openers); dev reserved for
validation, test never opened; no LLM.

**Nine intents, five attributes, ten boundary rules.** The intents are single-label
and mutually exclusive; the five attributes — `conversation_state`, `urgency`,
`frustration`, `language`, `social_nonrequest_indicator` — are orthogonal and
always recorded alongside. Boundary rules **BR-1…BR-10** resolve the confusable
pairs.

**The taxonomy is designed, not discovered.** KMeans silhouette over TF-IDF
openers is ≈0 at every k tested (best 0.0125 at k=10, several negative). There is
no natural cluster structure in short support text. What the evidence *does*
support is that the classes predict brand behaviour: historical DM-deflection
ranges 11.7%→77.5% across NMF topics, stable at k = 6, 8 and 10.

**Nine proposed intents** — `account_access`, `billing_subscription`,
`plans_eligibility`, `content_availability`, `playback_playlist`,
`app_device_technical`, `product_feature_feedback`, `social_praise`,
`other_unclear`.

**Decisions encoded:**

- **`help / need / asap` is not an intent.** It is an urgency register, not a
  subject: at k=10 it split into two near-identical pleading clusters (DM 56.4%
  and 45.1%) whose content was account and billing issues. Urgency and
  frustration are attributes. No intent may be defined by emotion, urgency, or a
  generic request for help.
- **`playback_playlist` and `app_device_technical` stay separate** despite similar
  DM rates (14.9% vs 16.3%, flagged as a merge pair by the action-profile test).
  Their retrieval evidence and drafted replies differ — music logic versus
  device/version troubleshooting. Similar escalation behaviour is not sufficient
  grounds to merge. BR-4 governs the boundary.
- **Conversational state is an attribute, not an intent.**
  `existing_case_followup` is the most predictive single signal measured
  (reply-DM 94.9% vs 36.7%) but occurs in 0.41% of openers — unlearnable as a
  class, valuable as a flag. `explicit_question` was tested and **rejected
  outright**: 41.97% prevalence, reply-DM 34.3% vs 38.8%, no operational signal.
  `social_praise` survives as an intent because it changes required behaviour —
  acknowledge, and run no retrieval at all.
- **`language` is an attribute (added in codebook 0.1.1), never an intent and
  never on its own an escalation reason.** Pilot sampling surfaced Indonesian,
  Swedish and Thai messages in 3 of 40 items (7.5%), far above the 0.2% character
  heuristic in `docs/profile.md` that that document already flags as
  untrustworthy. A non-English message with clear intent is classified normally
  and escalated on the same policy as any other; only genuinely undeterminable
  intent becomes `other_unclear`, with `language` recording why.
- **A ninth intent, `product_feature_feedback`, is proposed** beyond the original
  spine. Justification is operational, not statistical: nothing can be fixed and
  nothing looked up, so the correct reply is categorically different. Folding it
  into a technical intent would put unfixable requests into a troubleshooting
  retrieval pool and invite drafts promising fixes that will never ship — a
  concrete hallucination risk.

**Escalation authority.** `account_access` (MUST proxy 23.0%),
`billing_subscription` (11.2%) and `other_unclear` may force `ESCALATE`
independently. All others inherit the D9 policy. Note `plans_eligibility`:
reply-DM 65.9% but MUST proxy 0.2% — the brand DMs to look up an account, not
because the case is dangerous. Merging it into billing on DM similarity would
import a false escalation signal. This is the clearest evidence for D9's
separation of behaviour from policy.

**Retained rules.** ≥4% prevalence floor (≈8 examples in a 200-item core); merge
where pilot mutual confusion >25% *unless* escalation outcomes differ; split only
where the distinction changes what the system does.

**BR-10 — topic without an actionable request** (added in codebook 0.1.1). A
message that names a topic but states no request, problem, question or actionable
need is `other_unclear`; *"question about billing."* is **not**
`billing_subscription`. Deliberately strict: inferring the unstated issue
manufactures labels the text does not support. Artist/music distribution
(creator-side) is recorded as an **open coverage gap** handled as `other_unclear`
— no `creator` or `artist_distribution` intent has been added.

**Open / at risk.** `product_feature_feedback` measures 3.63% under a narrow seed
and `social_praise` may fall below 4% once polite requests are excluded — both
must clear the floor in the pilot. Prevalence figures are ranges bracketing two
disagreeing proxies (NMF over-assigns, keyword seeds under-assign); the pilot
produces the real numbers. Falsification criteria are listed in
`docs/taxonomy.md` §6.

## D18 — Temporal split **[MEASURED]**

Split on **opener date**; whole threads follow their opener.

| Split | Window | Spotify openers |
|---|---|---|
| train | 2017-10-01 → 11-05 | 14,547 |
| dev | 2017-11-06 → 11-19 | 6,716 |
| test | 2017-11-20 → 12-03 | 6,808 |

Assigning whole threads to the opener's window makes boundary-straddling
**structurally impossible** — measured 0, against 288 under the previous
per-tweet definition. Leakage controls retained: no self-retrieval, no
same-thread retrieval, customer-disjoint, near-duplicates removed.

~99.5% of Spotify's data falls in this nine-week span, so there is little room
for a materially different split.

## D19 — Cost: target ₹0–₹500, ceiling ~₹1,000 **[POLICY]**

Aggressive caching of every LLM call, keyed on `(model, prompt_hash, params)`,
written to disk. No paid infrastructure of any kind.

Forced by the environment: of the LLM providers, only `api.anthropic.com` is
reachable from this machine — Kaggle, Hugging Face, OpenAI, Google and
OpenRouter egress are all blocked. "Free-tier or local generation" from the
previous plan is therefore not available, and with no GPU and 3 GB RAM local
inference is impractical. One pinned Anthropic model for generation and a pinned
stronger Claude for judging, both cached, fits comfortably inside the ceiling.

## D20 — Recorded limitation: same-family judge bias **[OPEN]**

Because only Anthropic is reachable (D19), generator and judge come from the same
model family. LLM judges are known to favour outputs from their own family. This
cannot be engineered away with one provider available.

Mitigations: a different (stronger) model tier for judging than for generation;
D11's human-agreement validation; D11's corruption probes. Disclosed explicitly
as a limitation in the report rather than left for a reviewer to notice.

## D21 — Human pilot before the golden set **[POLICY]**

~40 blind items labelled before any golden-set labelling. Label intent,
escalation, reason, attributes, and genuine uncertainty. **Do not look at the
historical brand reply while labelling** — it would anchor the policy label onto
observed behaviour, collapsing D9.

Hesitation and confusion during the pilot drive codebook revision; then the
codebook is locked. Never adjust human labels to improve a system's numbers.

Pilot prerequisites are brand lock (D1), a drafted codebook and a sampling
script — not a working agent. **Gated on explicit go-ahead.**

## D22 — Reproducibility cache **[OPEN]**

The assignment requires headline results reproducible in <15 minutes from a
clean clone. A grader without an API key can only achieve that if the LLM
response cache ships **in the repository**, which conflicts with ignoring
generated caches.

Currently ignored, with the carve-out documented in `.gitignore`
(`!cache/llm/`). Must be settled before submission: either commit a small
pinned cache, or provide a subsample path that runs inside the budget.

## D23 — No human-labelled training set: every classifier is labelled by construction **[POLICY]**

**The project has no human-labelled training or dev set.** The 200 golden items
are the only human labels that exist, and they are TEST-window items. D7 fixes
that the test window is touched once, for final numbers, so those labels cannot
be used to fit, tune, threshold or model-select anything.

The direct consequence, stated plainly because it changes how every intent number
in the report must be read:

> **The conventional supervised baseline named in D14 — TF-IDF + logistic
> regression trained on human labels — cannot be built.** There is no labelled
> training data to fit it on, and fitting it on the golden set would be the exact
> contamination D7 exists to prevent. Any "supervised accuracy" reported for this
> project would be fabricated. None is reported.

What replaces it, and how each is constructed:

| Baseline / system | How its labels or rules were obtained | Sees golden labels? |
|---|---|---|
| **Seed-rule intent** (`rule`) | First-match-wins regex families from `scripts/taxonomy.py`, written against **train** text before any labelling | never |
| **Distant-supervision intent** (`distant-lr`) — *proxy baseline, not a supervised baseline* | TF-IDF + logistic regression fitted on **train openers weakly labelled by those same seed rules**. Human labels are absent from training | never |
| **LLM few-shot intent** (`llm`) | Zero/few-shot prompting with the locked codebook and **codebook examples only** (those examples are train openers, `docs/taxonomy.md` §2); prompts developed by inspection on **dev** | never |
| **Escalation: codebook lookup** (`lookup-codebook`) | `escalation_role` field in `docs/codebook.json`: `independent` → ESCALATE, `inherits` → AUTO_OK | never |
| **Escalation: always / never** | Constant | never |
| **Escalation: oracle lookup** (`lookup-oracle`) | Each intent mapped to its **majority escalation decision in the golden set itself** | **YES — in-sample** |

**`distant-lr` is a proxy baseline and must be labelled as one everywhere.** It is
not a supervised model in the usual sense: its training targets are regex output,
so it inherits every bias of the seeds and can be no better than them except
where TF-IDF generalises the vocabulary. Reporting it as "TF-IDF + LR" without
that qualifier would imply human supervision that does not exist. Two assumptions
are being made and neither is verified: that seed-family membership is a usable
stand-in for intent, and that the train and test windows are similar enough for a
train-fitted vectoriser to transfer. D16 already measures one distribution shift
between those windows, so the second assumption is known to be imperfect.

**`lookup-oracle` is fitted on the evaluation labels and is therefore not a fair
baseline.** It is reported as a **ceiling**: the score an escalation model would
reach if it perfectly learned the intent and nothing else. It exists because the
golden set shows intent very nearly determines escalation (93.5% in-sample,
`docs/golden_results.md` §7.1), and a system beating `lookup-codebook` while
merely approaching `lookup-oracle` has demonstrated intent classification, not
safety judgement. It is always printed with its in-sample warning attached.

**Hyperparameters are fixed a priori, not searched.** With no labelled dev set
there is nothing to select on, so `distant-lr` uses documented defaults and no
sweep is run. This is a limitation, not a tuning result, and is recorded as such.

**Rejected: cross-validation inside the golden set.** Fitting and reporting on the
same 200 items contaminates the only evaluation the project has.

---

## Main methodological risks

1. Template reuse inflating apparent reply quality (21.4% verbatim reuse) — D13
2. Unvalidated LLM judging, worsened by same-family bias — D11, D20
3. Treating historical DM behaviour as escalation truth — D9, D16
4. Threshold tuning against the test set — D7
5. Biased stress sampling leaking into headline numbers — D6
6. Customer/thread overlap and near-duplicates across the split — D18
7. A headline number too underpowered to support its claim — D4, D7
