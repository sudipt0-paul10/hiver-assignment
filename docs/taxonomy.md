# SpotifyCares Intent Codebook — PROPOSED

**STATUS: version 0.2.1-proposed. Pilot locked; taxonomy frozen for golden-set labelling.**

*0.2.0 is the post-pilot revision, driven by the 40-item pilot: `social_praise`
is **removed** (9 intents → 8) and folds into `other_unclear`; BR-11 is added;
BR-4 is amended; BR-5 is repointed. The remaining eight intent names and their
core boundaries are unchanged from 0.1.1. Evidence is in §8.*

Derived from the **train split only** (2017-10-01 → 2017-11-05, 14,547 openers).
Dev is reserved for validating this taxonomy; the test window has never been
opened. No LLM was used at any point.

Reproduce the evidence:

```bash
python scripts/taxonomy.py --csv twcs/twcs.csv --k 8      # structure + action profiles
python scripts/taxonomy.py --csv twcs/twcs.csv --dump-examples 14   # real example pool
```

Machine-readable form: `docs/codebook.json`.

---

## 0. What the evidence does and does not support

**The taxonomy was not discovered; it was designed.** KMeans silhouette over
TF-IDF openers is ≈ 0 at every k tested (best 0.0125 at k=10, several negative).
There is no natural cluster structure — support tweets are short and share
vocabulary. Anyone claiming to have "found the latent intents" in this data is
overselling.

**What the evidence does support is that these classes predict what the brand
did.** Historical DM-deflection across NMF topics ranges 11.7% → 77.5%, a 6.6×
spread, stable across k = 6, 8 and 10. That is the criterion D17 actually sets:
split only where the distinction changes what the system does.

So every class below is justified on four grounds — prevalence, operational
distinction, retrieval/reply distinction, escalation implication — and the
proposal is falsifiable by the pilot.

### Two measurement caveats that bound everything here

1. **Prevalence figures are estimates from two disagreeing proxies**, given as
   ranges. NMF soft-topic shares over-assign (assignment is noisy when silhouette
   ≈ 0); keyword seeds under-assign (they left 60.2% of train in `other_unclear`,
   and inspection shows most of that residue *is* assignable — "when will spotify
   come to more countries", "i canceled my account and you still bill me"). The
   truth sits between. **The pilot produces the real numbers.**
2. **MUST figures are the keyword proxy from `profile.MUST_PROXY`**, not labels.

All examples below are **real train openers**, shown after D10 normalisation
(mentions and URLs stripped, lowercased) — which is why they appear lowercase.
Quotes preserve the source wording exactly; mentions and URLs are absent because D10 normalisation removed them. One example carries `[name redacted]` where a third party was named. See §5.

---

## 1. Attributes — defined separately from intent

Attributes are **not** intent values and never compete with them. Every item gets
exactly one intent *and* a value for each attribute.

| Attribute | Values | Definition |
|---|---|---|
| `conversation_state` | `opener` · `existing_case_followup` · `unclear` | Whether this message starts a new issue, continues one already in a private channel, or cannot be told apart. |
| `urgency` | `low` · `normal` · `high` | How time-critical the customer asserts the issue is. Asserted, not adjudicated. |
| `frustration` | `low` · `normal` · `high` | Emotional intensity directed at the brand/product. |
| `social_nonrequest_indicator` | `true` · `false` | Politeness, thanks or praise language is present. **Independent of whether the item is a request.** |
| `language` | `english` · `non_english` · `unclear` | The language the message is written in. A property of the message, not of what the customer wants. |

### Why these are attributes, not intents — measured

- **`existing_case_followup` is the single most predictive signal in the data**
  (DM-followup language: reply-DM 94.9% vs 36.7% when absent) **but occurs in only
  0.41% of openers.** Far below the 4% viability floor. As a class it would be
  unlearnable and unmeasurable; as a flag it is valuable. This is precisely why it
  is an attribute.
- **Urgency and frustration do not identify a topic.** The pilot's
  `help / need / asap` meta-cluster had a 58% DM rate, which looks like signal —
  but at k=10 it split into two near-identical pleading clusters (DM 56.4% and
  45.1%), and its examples are account and billing problems wearing urgent
  language: *"i need help with my account i keep getting charged for an account
  that is on the free subscription"*. It was a register, not a subject.
- **`explicit_question` was tested and rejected entirely** — 41.97% prevalence but
  reply-DM 34.3% vs 38.8% when absent. No operational signal. Not an attribute,
  not an intent, dropped.
- **`social_nonrequest_indicator` carries the praise signal on its own** — since
  0.2.0 there is no `social_praise` intent — because politeness co-occurs with
  real requests constantly. The
  keyword seed for praise pulled in *"my daily mix keeps stopping for no reason.
  it's not my internet connection. help please. thanks."* — an
  `playback_playlist` request that happens to be polite.

- **`language` is an attribute, never an intent and never an escalation reason
  on its own.** Pilot sampling surfaced Indonesian, Swedish and Thai messages in
  3 of 40 items (7.5%) — far above the 0.2% character heuristic in
  `docs/profile.md`, which that document already flags as untrustworthy. The
  rules:
  - If a non-English message has **sufficiently clear intent, classify it
    normally.** Being in another language does not make an item unclear, and does
    not by itself justify escalation.
  - If intent **cannot be determined reliably** — whether for language reasons or
    any other — the intent is `other_unclear`, and `language` records why.
  - `unclear` is for mixed, transliterated, or too-short-to-tell cases.
  - Escalation is decided on the same policy as any other message. `language` is
    never an `escalation_reason`.

Sarcastic thanks exists but is negligible (0.03%, n=5), e.g. *"thanks for ruining
my night"*. Not worth a rule; annotators judge intent by the request, not the
politeness token.

---

## 2. Proposed intents

Eight classes, after the pilot removed `social_praise` (§8).

Summary — historical action profile from train (`reply DM` = share of first brand
replies that deflect to DM; `MUST` = keyword proxy):

| # | Intent | Est. prevalence | reply DM | MUST proxy | Escalation role |
|---|---|---|---|---|---|
| 1 | `account_access` | 10–15% | **76.9%** | **23.0%** | **independent** |
| 2 | `billing_subscription` | 6–9% | 70.6% | 11.2% | **independent** |
| 3 | `plans_eligibility` | 5–8% | 65.9% | 2.3% | inherits |
| 4 | `content_availability` | 5–8% | 11.7% | 0.1% | inherits |
| 5 | `playback_playlist` | 20–30% | 14.9% | 0.6% | inherits |
| 6 | `app_device_technical` | 10–14% | 16.3% | 1.3% | inherits |
| 7 | `product_feature_feedback` | 4–8% *(prov.)* | 15.3% | 0.4% | inherits |
| 8 | `other_unclear` | 5–12% | 34.4% | 0.3% | **independent** |

### 2.1 Measured vs projected prevalence — correction

**The "Est. prevalence" column above is superseded as a source of prevalence
figures.** Those ranges are pre-label proxy estimates (NMF soft-topic shares and
keyword seeds), kept here unchanged for historical and audit purposes. The
160-item unbiased core of the golden set is the first measurement from human
labels, and two of the eight projections are not supported by it.

Measured = golden-set core, n=160, Wilson 95%:

| Intent | Projected | Measured (core, n=160) | Verdict |
|---|---|---|---|
| `account_access` | 10–15% | 14.4% [9.8, 20.6] | consistent |
| `billing_subscription` | 6–9% | **16.9% [11.9, 23.4]** | **not supported** — interval excludes the projection |
| `plans_eligibility` | 5–8% | 8.1% [4.8, 13.4] | consistent |
| `content_availability` | 5–8% | 7.5% [4.3, 12.7] | consistent |
| `playback_playlist` | 20–30% | **8.8% [5.3, 14.2]** | **not supported** — interval excludes the projection |
| `app_device_technical` | 10–14% | 18.8% [13.5, 25.5] | consistent, but only just (overlap 13.5–14.0) |
| `product_feature_feedback` | 4–8% | 10.0% [6.2, 15.6] | consistent |
| `other_unclear` | 5–12% | 15.0% [10.3, 21.3] | consistent |

Two consequences follow directly:

1. **`playback_playlist` is not "the largest class".** The characterisation in §2
   and in §5 H/J — largest class, "the volume class, and therefore where
   auto-handle coverage is won" — rests on the 20–30% projection and is **not**
   supported by the measured sample. That wording is left in place unedited; read
   it as superseded by this section.
2. **The measured order is different.** By core count the largest classes are
   `app_device_technical`, `billing_subscription`, `other_unclear` and
   `account_access` — not playback.

**Why the proxies missed, stated as a finding and not a proven cause.** The
pattern across all eight rows is that the proxy construction appears to have
over-weighted `playback_playlist` relative to the account, billing and device
families. That is an *observed* direction of error, consistent with §0's warning
that NMF assignment is noisy when silhouette ≈ 0 and that the seeds
under-assign. It is **not** a demonstrated causal explanation: no experiment
here isolates which of the seed vocabulary, the NMF topic assignment, or the
brand-reply-only pool produced it, and none is planned. Treat it as a caveat on
proxy-derived prevalence generally, not as a diagnosis.

**What this correction does not change.** The eight intents, their definitions,
inclusion and exclusion criteria, boundary rules BR-1 to BR-11, escalation roles,
and the golden-set sampling methodology are all unaffected — prevalence was never
the basis for any of them (§0: every class is justified on operational
distinction, not frequency). The measured figures also do not retroactively
change the pilot or the sampling design, which used the projections only to size
non-proportional strata that were never meant to estimate prevalence.

**Full analysis, subset breakdowns, escalation results and limitations:**
`docs/golden_results.md`. Do not quote the projected ranges as results.

---

### 1. `account_access`

**B. Definition.** The customer cannot get into, or has lost control of, their
account — including credential failure, lockout, and compromise.

**C. Inclusion.** Login or sign-in failure; forgotten/incorrect password;
email or username changed without consent; account hacked, hijacked, disabled or
suspended; third-party login (Facebook) blocking access; requests to delete or
recover an account; someone else using the account.

**D. Exclusion.** Can log in but disputes a charge → `billing_subscription`. Can
log in but the plan is wrong → `plans_eligibility`. "Account" used loosely to
mean "my Spotify" with no access failure → classify by the actual problem.

**E. Real examples.**
1. *"still can't access my account as someone has hacked it. if my playlists have gone i'll be devastated"*
2. *"login button on the app doesn't work, help!!"*
3. *"2nd time in as many weeks has someone been able to change the email address on my account and then reset the password"*
4. *"my account was hacked and now i can;t log in. please help! i use spotify everyday!"*
5. *"i can't log in on any device and i've reset my password twice"*
6. *"checking in w/ because someone apparently changed my account email address and now i can't log in. i emailed but no response"*
7. *"i no longer have facebook so i cant access contact form, community, or login. i want to delete my account. please advise asap"*
8. *"i am having trouble logging in to my account. i can't login with fb, my email or anything. i have a premium account."*
9. *"cant login to spotify on xbox one if you login via facebook? keeps telling me to change my fb password, why?"*

**F. Boundary cases.**
- *"my premium account got hacked. luckily i was able to change my password and delete other apps. what are you doing to fix this?"* → **`account_access`**. "Premium" is incidental; the subject is compromise.
- *"hi, i bought a premium subscription but when i log in to the app i still have a free account"* → **`billing_subscription`**. They *can* log in; the entitlement did not apply. The word "log in" is a decoy.
- *"yo y'all need to fix the facebook integration, nothing happens when i try to login to facebook on spotify 1.0.66.478 windows 10"* → **`account_access`**. Reported as a platform bug, but the outcome is inability to authenticate. If login succeeds and something else breaks, it is `app_device_technical`.

**G. Most confusable with.** `billing_subscription`, `app_device_technical`, `plans_eligibility`.

**H. Prevalence.** 10–15% (NMF 13.4–15.7%; narrow seed 5.4%).

**I. Escalation.** **Independent.** Carries by far the highest escalation mass
(MUST proxy 23.0%, reply-DM 76.9%). Account compromise and lockout cannot be
resolved in public and touch security. This intent alone should be able to force
`ESCALATE` regardless of other signals.

**J. Operational value.** Retrieval evidence is a distinct, near-templated family
("can you DM us your account's email address?"), and the correct output is a
deterministic handoff, never a generated fix. Keeping it separate is what makes
the deterministic-handoff path in D8 targetable.

---

### 2. `billing_subscription`

**B. Definition.** Money has moved, or failed to move, and the customer disputes
or questions it.

**C. Inclusion.** Charged twice/unexpectedly/after cancelling; wrong amount;
refund requests; failed or declined payment; cancellation that did not take
effect; billing date changes; paid but entitlement not received.

**D. Exclusion.** Asking what a plan costs or whether they qualify, with no
charge in dispute → `plans_eligibility`. Cannot log in → `account_access`.

**E. Real examples.**
1. *"i got charged twice in september. can you please help refund this money back. thanks"*
2. *"hey! i noticed that you guys charged me 10.79 instead of applying the student discount. why might that be?"*
3. *"just got charged for premium, i was meant to cancel before it charged me so i've just canceled right now can i get a refund??"*
4. *"i am being charged for premium but it still says i have a 'free' account. tried all the tricks recommended. want money back."*
5. *"you charged me six times last night because i was reloading the page that claimed my card *wasn't* being accepted"*
6. *"can someone please explain why you are taking double payments each month i have been getting charged the last four months???"*
7. *"help i cancelled my accounts back in august i've just notice that i'm still being charged"*
8. *"i just upgraded my account and my account was charged but i don't have premium???????"*
9. *"(1/2) just been charged for another month however i have a free subscription with my mobile phone contract!"*

**F. Boundary cases.**
- *"looks like my playlist was cancelled frm a missed pymt i'll pay now so pls put songs back"* → **`billing_subscription`**. Mentions playlists, but the cause and fix are a missed payment. Retrieved by the playback seed — a genuine trap.
- *"yo spotify, i'm a student but i'm still being charged $9.99 per month"* → **`billing_subscription`**. Student discount appears, but a specific wrong charge is disputed. Contrast with example 3 under `plans_eligibility`.
- *"like to change my billing date but due to be billed 2mro. should i wait & pay before i cancel?"* → **`billing_subscription`**. Forward-looking and nothing is wrong yet, but the subject is the payment mechanism.

**G. Most confusable with.** `plans_eligibility`, `account_access`.

**H. Prevalence.** 6–9% (NMF 7.3–8.3%; seed 5.2%).

**I. Escalation.** **Independent.** Reply-DM 94.8% under the seed view — the
highest of any class — and MUST proxy 11.2–32.2% concentrated in
`payment_dispute`. Financial disputes require account access and carry
regulatory weight; they must be able to force `ESCALATE` on their own.

**J. Operational value.** Historical replies are a distinct family requesting
private details; a public generated reply about someone's money is exactly the
unsafe-auto case the headline metric (D3) is built to catch.

---

### 3. `plans_eligibility`

**B. Definition.** Questions about which plan applies, joining/switching plans, or
qualifying for a price — with no specific charge in dispute.

**C. Inclusion.** Family plan invitations and membership mechanics; student
verification and discounts; free trials, promos and first-month offers; upgrading
or switching plans; gift cards; "do I qualify".

**D. Exclusion.** A concrete wrong charge or refund → `billing_subscription`.
Cannot access the account at all → `account_access`. Asking for a plan that does
not exist → `product_feature_feedback`.

**E. Real examples.**
1. *"i got the student discount and it went through successfully, but it keeps going away and putting me at the normal rate"*
2. *"please help. signed up to a premium family account but the email is not going to my wife."*
3. *"please i'm having problems to accept the invittion to join the family membership since monday."*
4. *"i just signed up for an account and was trying to get the 99p for 1st month offer but instead it's done a week free"*
5. *"hi! i'm trying to upgrade my account (i was previously in a family plan, but not anymore) and it's not accepting my cc?"*
6. *"still can't activate all invited member of my premium family..."*
7. *"i was in a spotify family plan for about week, but it suddenly stopped giving me premium and i cant enter the plan anymore."*
8. *"i am trying to take advantage of the 30day free trial but when i click through it doesn't allow sign up"*
9. *"i'm having trouble using a nus card for discount :( please help"*

**F. Boundary cases.**
- *"i've been unknowingly paying full price for spotify since march"* → **`billing_subscription`**. Eligibility is the backdrop, but money was wrongly taken over months. The dispute governs.
- *"why do only college students get a discount? what about high school students? we enjoy music too..."* → **`product_feature_feedback`**. No eligibility question about *this* customer; it argues the policy should change.
- *"dear can you make some sort of family plan so my husband and i can stop arguing via song titles"* → **`product_feature_feedback`**. Requests a plan that does not exist.
- *"i baught the wrong giftcard, ofcourse :). i have premium family and got premium. it says it doesnt match my current status."* → **`plans_eligibility`**. Money was spent, but the problem is plan mismatch, not a disputed charge.

**G. Most confusable with.** `billing_subscription`, `product_feature_feedback`.

**H. Prevalence.** 5–8% (NMF 6.5–7.0%; seed 3.4%).

**I. Escalation.** **Inherits.** The sharpest finding in this codebook: reply-DM
is high (65.9–73.6%) but MUST proxy is **0.2%** — two orders of magnitude below
`account_access`. The brand DMs because it needs to look up an account, not
because the issue is dangerous. Merging this into `billing_subscription` on DM
similarity would import a false escalation signal.

**J. Operational value.** This is the class that most justifies separating
*behaviour* from *policy* (D9). Distinct retrieval family (family-plan invite
mechanics, student verification steps) and many cases are genuinely
auto-answerable with a public how-to.

---

### 4. `content_availability`

**B. Definition.** Content, or the service itself, is absent from the catalogue or
the customer's region.

**C. Inclusion.** Album/track/podcast not on Spotify; content removed or taken
down; release-date questions; regional catalogue gaps; **service not available in
a country** (see rule BR-7).

**D. Exclusion.** Content exists but will not play for this user →
`playback_playlist`. A product *feature* was removed → `product_feature_feedback`.

**E. Real examples.**
1. *"bruh y'all removed dreams and gasoline by rob baird and the entire album of deadmau5's get scraped, what's the deal?"*
2. *"please release got7 '7 for 7' album"*
3. *"hi! when will u upload btob's new album brother act"*
4. *"yo are you gonna make spotify available in kuwait anytime soon?"*
5. *"when will the ep be on the app?"*
6. *"what's happened with all of music, some of #davidcrowderband, and lots of others the past few weeks? not available!"*
7. *"a few of my fav songs got taken off of and i'm super sad cause i can't play them on repeat anymore"*
8. *"why aren't you available in india.."*
9. *"hey there is a new album from called #replay , many great song choices and a great tribute to prince's #purplerain"*

**F. Boundary cases.**
- *"still can't understand why you removed the touch preview feature? made it so easy to build new playlists, bring it back pls!"* → **`product_feature_feedback`**. "Removed" is present, but what was removed is a *feature*, not content. Retrieved by the availability seed — a real trap.
- *"a few of my fav songs got taken off of and i'm super sad cause i can't play them on repeat"* → **`content_availability`**. Contains "can't play", but the cause is catalogue removal, not playback failure.
- *"i might actually listen to release radar if you filtered out remixes"* → **`product_feature_feedback`**. Mentions a release feature but requests behaviour change.

**G. Most confusable with.** `playback_playlist`, `product_feature_feedback`.

**H. Prevalence.** 5–8% (NMF 5.5–5.8%; seed 3.8%).

**I. Escalation.** **Inherits.** Lowest-risk class measured: reply-DM 11.7%,
MUST proxy 0.1%. Prime auto-handle territory.

**J. Operational value.** The historical reply family is highly consistent
("there's info about Spotify content here…" was the #2 template overall at 142
occurrences), making it the strongest retrieval case in the dataset and a good
candidate for demonstrating that retrieval-grounded drafting works.

---

### 5. `playback_playlist`

**B. Definition.** Music that exists does not play, or playlist/library behaviour
is wrong — independent of which device is used.

**C. Inclusion.** Songs won't play, skip, pause or stop; shuffle and repeat
behaviour; queue; offline downloads disappearing; playlist creation, ordering,
sharing, collaborative playlists; library/saved music loss; recommendations and
Discover Weekly quality.

**D. Exclusion.** The client app crashes or won't update → `app_device_technical`.
The content isn't in the catalogue → `content_availability`. A missing capability
is requested → `product_feature_feedback`.

**E. Real examples.**
1. *"i had niall horan's album on shuffle when all of a sudden taylor swift started playing ????? get it together plz"*
2. *"all my downloaded songs were gone. my sd card wasn't corrupted, have visited spotify w/in 30 days..."*
3. *"my downloaded albums get deleted at least once a month. seems to correlate to your updates."*
4. *"i accidentally hit the thumbs up on bubbly and now i can never listen to john mayer's spotify radio station ever again."*
5. *"so wait, , there's no way to add a someone on spotify to a playlist through spotify? i have to use a third party?"*
6. *"disappointed there was no #seal on #tbt animal playlist"*
7. *"hi i just got a new laptop &amp; wondering if there's a way to transfer downloaded songs from my old one w/o redownloading? thanks"*
8. *"my daily mix keeps stopping for no reason. it's not my internet connection. help please. thanks."*
9. *"i'm pissed my shuffle and repeat button just don't fucking work and i'm getting frustrated"*

**F. Boundary cases.**
- *"i can play several albums, including this one in chrome, but the application won't play it"* → **`app_device_technical`**. The customer has isolated it to one client; the fault is the app, not playback generally. Rule BR-4 applies.
- *"my downloaded albums get deleted at least once a month. seems to correlate to your updates"* → **`playback_playlist`**. Blames updates, but the lost thing is offline music. Contrast with the previous case.
- *"looks like my playlist was cancelled frm a missed pymt"* → **`billing_subscription`**. See BR-1.
- *"needs an option within a playlist to only play non explicit music"* → **`product_feature_feedback`**. Playlist subject, but the ask is a new capability.

**G. Most confusable with.** `app_device_technical`, `content_availability`, `product_feature_feedback`.

**H. Prevalence.** 20–30% (NMF 25.0–31.2%; seed 12.3%). **The largest class.**

**I. Escalation.** **Inherits.** reply-DM 14.9%, MUST proxy 0.6%.

**J. Operational value.** The volume class, and therefore where auto-handle
coverage is won. Retrieval evidence is troubleshooting steps; replies are
public how-tos. Explicitly kept separate from `app_device_technical` per the
brief, despite similar DM rates — see BR-4 for why.

---

### 6. `app_device_technical`

**B. Definition.** The Spotify client, or its integration with an OS, device or
third-party hardware, malfunctions.

**C. Inclusion.** App crashes, freezes, won't install or update; version-specific
bugs; OS integration (iOS, Android, Windows, Mac, Xbox); wearables and Apple
Watch; connected devices — Chromecast, Alexa, Sonos, CarPlay, Bluetooth; web
player faults; battery and performance; connectivity errors.

**D. Exclusion.** Music behaviour that is wrong on every surface →
`playback_playlist`. Login failure → `account_access`. A requested capability →
`product_feature_feedback`.

**E. Real examples.**
1. *"why my spotify can't connect to internet?? 😭 other apps working just fine 💔"*
2. *"i've tried multiple times to fix the issues but i cannot get the app or the web player to play anything."*
3. *"25% of iphone battery use has gone on 2.6hrs of background refresh for today. is there something i can switch off to stop this?"*
4. *"web player stuck on advertisement but wont play it. greyed out. rebooted and logged in and out already."*
5. *"wow, has a terrible bug on android. if you get logged out in offline mode you can't log back in without reinstalling the app."*
6. *"(1) using android on samsung galaxy a4. music keeps stopping and saying my account is in use elsewhere,"*
7. *"weird that spotify has a landscape mode on android, but not ios."*
8. *"i've had your premium service on windows phone for 3 or 4 days now. none of these days have i had no crashes."*
9. *"thanks for dumping my saved music when i updated my ios"*

**F. Boundary cases.**
- *"why is the playlist scroll sidebar on the right (like on android) missing on ios? scrolling down takes ages with my iphone"* → **`app_device_technical`**. Platform-specific UI defect, not a playlist-logic problem.
- *"disproportionately stressed at the fact that since i upgraded to ios 11 my earphone controls operate apple music rather than [spotify]"* → **`app_device_technical`**. OS integration regression.
- *"hey , why do you keep resetting album download radio button? i keep having to redownload. ios."* → **`playback_playlist`**. Names iOS, but the lost thing is offline downloads. BR-4's "where does the fix live" test decides it.

**G. Most confusable with.** `playback_playlist`, `account_access`.

**H. Prevalence.** 10–14% (NMF 10.6–12.0%; seed 6.3%).

**I. Escalation.** **Inherits.** reply-DM 16.3%, MUST proxy 0.0–1.3% — the lowest
MUST of any substantive class.

**J. Operational value.** Retrieval evidence is device- and version-specific
troubleshooting, materially different from playback how-tos. Drafts need to
reference platform and version; merging with `playback_playlist` would blur the
retrieval pool that makes those drafts correct. This is the class the brief's
decision 2 protects.

---

### 7. `product_feature_feedback`

**B. Definition.** A request for a capability that does not exist, or evaluative
feedback on product/design decisions, with no malfunction to fix.

**C. Inclusion.** Feature requests; "please add / bring back / there should be";
design and UX complaints about intended behaviour; policy arguments (who should
qualify for a discount); requests to launch in a market *as a product decision*
rather than a catalogue question.

**D. Exclusion.** Something is broken → the matching technical intent.
Praise with no request → `other_unclear` (BR-5).

**E. Real examples.**
1. *"can you please add lyrics back and allow us to put our playlists and songs in orders (abc, year, date added, etc.,)"*
2. *"feature request for mac app, profiles like work, home etc. i hate disabling/enabling proxy when at home/work"*
3. *"i wish there was a way to block songs on spotify so i wouldnt have to hear the songs i despise anymore"*
4. *"feature request: ability to switch to a new playlist after the current song (instead of stopping it and switching immediately)"*
5. *"can you please add a feature to only view from releases from artist you follow."*
6. *"can you please add a feature to easily view song lyrics within phone app?"*
7. *"'please add an equalizer for pc users' -my neighbors"*
8. *"needs an advertisement feedback option because i swear if i have to listen to another northwest nazarene ad i'm going to lose it"*
9. *"please please please don't deploy this design throughout the app. personally don't appreciate the changes"*
10. *"still can't understand why you removed the touch preview feature? bring it back pls!"*

**F. Boundary cases.**
- *"why can't i add a family member to my premium family plan ????"* → **`plans_eligibility`**. Phrased as a capability complaint, but family invites *do* exist, so this is a failing mechanic.
- *"there should be a hashtag trending which requests and in a way forces you to bring spotify to india"* → **`product_feature_feedback`**. Compare with *"why aren't you available in india.."* → `content_availability` (BR-7): a plain availability question versus advocacy.
- *"i might actually listen to release radar if you filtered out remixes"* → **`product_feature_feedback`**. Conditional praise wrapping a feature request.

**G. Most confusable with.** `playback_playlist`, `app_device_technical`, `plans_eligibility`, `content_availability`.

**H. Prevalence.** 4–8%. The narrow seed measured 3.63% (n=528), below the 4%
floor, which is why it was flagged provisional in 0.1.x. **The pilot resolved
this:** 7 of 40 labels, the second-largest class. No longer provisional.

**I. Escalation.** **Inherits.** reply-DM 15.3% vs 37.8% when absent; MUST proxy
0.38%. Cleanly auto-handleable.

**J. Operational value.** This is why it is proposed despite not being in the
agreed spine: **the correct reply is categorically different.** Nothing can be
fixed and nothing can be looked up — the right output acknowledges and routes to
product feedback. Folding it into `playback_playlist` or `app_device_technical`
would put unfixable requests into a troubleshooting retrieval pool and invite
drafts that promise fixes that will never come — a concrete hallucination risk.

**Resolved in 0.2.0:** adopted. The pilot cleared the floor comfortably.

---

### 8. `other_unclear`

**B. Definition.** The message cannot be confidently assigned — too little
information, genuinely ambiguous, off-topic, spam, or not interpretable.

**C. Inclusion.** Truncated or context-free messages; multi-issue messages with no
dominant intent; unrelated promotion; unintelligible or untranslatable text.
Also, explicitly:

- **Topic without an actionable request (BR-10).** The customer names a subject
  but expresses no request, problem, question or actionable need. *"question
  about billing."* is `other_unclear`, **not** `billing_subscription`. Do not
  infer the issue they probably meant.
- **Intent not reliably determinable for language reasons.** Record
  `language = non_english` or `unclear` alongside. A non-English message whose
  intent *is* clear does **not** belong here.
- **Unsupported domains**, currently artist/music distribution — see §6.
- **Social-only praise** carrying no actionable request (`social_praise` was
  removed as an intent in 0.2.0). Record `social_nonrequest_indicator = true`.

**D. Exclusion.** Do not use as a dumping ground for *hard but assignable* items.
The seed run left 60.2% of train here, and inspection showed most of it is
assignable — that was a measurement artefact, not a real residue.

**E. Real examples.**
1. *"please consider accepting #bitcoincash for payments - it's fast and secure, with low fees!"*
2. *"what's this about? i've restarted my system and the issue still occured"*
3. *"hi can u answer my dm please"*
4. *"hello help me please"*
5. *"help on my spotify card please"*
6. *"wheres twicetagram by twice"* (ambiguous: catalogue absence or search failure)
7. *"nvm"*-style abandonments and truncated *(cont)* messages
8. *"#spotifycares_about your money"* — commentary with no request

**F. Boundary cases.**
- *"hi can u answer my dm please"* → **`other_unclear`** as intent, but
  `conversation_state = existing_case_followup`. Shows attributes carrying
  information the intent cannot.
- *"when will spotfiy come to more countries?"* → **`content_availability`**, not
  unclear. Landed in the seed residue purely because of the typo.
- *"i canceled my account,, and you still automatically bill me?"* → **`billing_subscription`**. In the residue only because the seed lacked "bill me".
- *"question about billing."* → **`other_unclear`** by BR-10. The topic is unambiguous; the request is absent. Assigning `billing_subscription` would be inferring an issue the customer never stated.

**G. Most confusable with.** All of them, by construction.

**H. Prevalence.** 5–12%, expected near the low end once a human applies the
codebook rather than regexes.

**I. Escalation.** **Independent.** Inability to determine intent is itself a
reason not to act autonomously. Should bias toward `ESCALATE` — an unclear
message is exactly where a confident auto-reply is most likely to be wrong.

**J. Operational value.** Its rate is a **health metric for the taxonomy**. If the
pilot puts more than ~15% here, the taxonomy is inadequate and must be revised
before the golden set is labelled.

---

## 3. Boundary rules (BR)

Applied in order; the first that fires decides.

**BR-1 — Billing vs account/access.** Ask: *can the customer get into the
account?* No → `account_access`. Yes, and a specific charge/refund is disputed →
`billing_subscription`. The token "account" is not evidence; access failure is.
*Test: "i bought a premium subscription but when i log in i still have a free
account" → billing (login works).*

**BR-2 — Plans/eligibility vs billing.** Ask: *has money already moved
incorrectly?* Yes → `billing_subscription`. No — questions about price,
qualifying, joining, switching → `plans_eligibility`. *Test: "i've been
unknowingly paying full price since march" → billing. "i'm having trouble using a
nus card for discount" → plans.*

**BR-3 — Content availability vs playback.** Ask: *does the content exist in the
catalogue for this customer's market?* No → `content_availability`. Yes, but it
won't play correctly → `playback_playlist`. *Test: "a few of my fav songs got
taken off and i can't play them on repeat" → availability, despite "can't play".*

**BR-4 — Playback/playlist vs app/device technical.** Ask: *where would the fix
live — in playback/catalogue behaviour, or in the client/OS/device?* Decisive
signal: if the customer reports it working on one surface and failing on another,
it is `app_device_technical`. If it fails everywhere, or concerns music logic
(shuffle, queue, offline, playlists), it is `playback_playlist`. Merely naming a
device does not move it. *Test: "i can play several albums in chrome, but the
application won't play it" → app/device. "why do you keep resetting album
download radio button? ios" → playback.*

**Device, OS and app names are context, not evidence:** they locate the symptom
and never by themselves assign the intent. If no client-specific surface is
mentioned, resolve the case under the relevant non-device rule rather than
assigning `app_device_technical` merely from generic wording. *Test: "am i the
only one who can't listen seventeen hello on spotify?" names no surface → resolve
under BR-3, not app_device_technical.*

**BR-5 — Social/praise vs an actual support request.** Ask: *is there anything to
act on?* Any request or question → that request's intent, with
`social_nonrequest_indicator = true`. Only if nothing is actionable →
`other_unclear` (the `social_praise` intent was removed in 0.2.0; the indicator
attribute carries the praise signal). *Test: "my daily mix keeps stopping… help please. thanks." →
playback_playlist + indicator.*

**BR-6 — Feature feedback vs a malfunction.** Ask: *does the capability exist?*

- **Exists and is malfunctioning** → the matching technical intent
  (`app_device_technical` for client/OS/device faults, `playback_playlist` for
  music logic).
- **Does not exist**, or the customer asks for it to be built, added, restored, or
  **extended to a new device or platform** → `product_feature_feedback`.

A request for support on a device or OS version that is **not yet supported** is a
request for a capability that does not exist, and is therefore
`product_feature_feedback`, **not** `app_device_technical`.

*Tests: "any idea when you'll update the app to support the iphone x" →
product_feature_feedback (unshipped). "your latest update doesn't work on iphone 5
with ios 10.3.3. endless app load screen" → app_device_technical (shipped, broken).
"why can't i add a family member" → plans_eligibility (invites exist, mechanic
failing). "needs an option to only play non explicit music" → feature feedback.*

**BR-7 — Market/region availability.** Service-not-in-my-country as a plain
question → `content_availability` (the answer is the same catalogue/roadmap
response). Framed as advocacy or a campaign → `product_feature_feedback`. *Test:
"why aren't you available in india.." → availability. "there should be a hashtag
trending which… forces you to bring spotify to india" → feedback.*

**BR-8 — Other/unclear vs a known intent.** `other_unclear` requires that a
careful reader **cannot** identify the issue — not that it is hard. If a
plausible intent is recoverable from the text, assign it and mark
`urgency`/`frustration` normally. Multi-issue messages take the intent of the
issue the customer leads with.

**BR-11 — Account access vs app/device technical.** If the customer cannot
authenticate — cannot log in, is unexpectedly logged out, cannot reset a
password, or the account is compromised — the intent is `account_access`,
**regardless of any phone, OS or app context**. Device/OS/app mentions describe
*where* the failure occurred, not *what* failed. `app_device_technical` applies
only when authentication is not the underlying issue. *Test: "iphone 5s and ios
8.1 / spotify app auto logged me out... still says 'log in failed'" →
account_access.*

**BR-10 — Topic without an actionable request.** If the message only identifies a
topic and expresses no request, problem, question or actionable need, assign
`other_unclear`. **Do not infer the customer's intended issue.** *Test: "question
about billing." → other_unclear, not billing_subscription.* This rule is
deliberately strict: guessing the unstated issue is how an annotator quietly
manufactures labels the text does not support.

**BR-9 — Multi-intent tie-break.** When two intents genuinely apply, prefer the
one with **independent escalation authority** (`account_access` >
`billing_subscription` > `other_unclear` > all others). Safety beats topical
neatness.

---

## 4. Annotation reliability notes

- Intent is **single-label and mutually exclusive**. Attributes are orthogonal and
  always recorded.
- Annotate from the **customer message only**. The historical brand reply must not
  be visible during labelling (D21) — it would collapse policy into observed
  behaviour (D9).
- The expected hard pairs, from the boundary analysis, are
  `billing ↔ plans`, `playback ↔ app_device`, and `feature_feedback ↔` everything
  technical. D17's merge rule applies: if pilot mutual confusion exceeds 25% and
  escalation outcomes do not differ, merge.
- `account_access ↔ billing` should *not* be merged even if confusable: their MUST
  proxies differ by roughly 2× and both hold independent escalation authority.

## 5. Handling of real customer text

Every example is a real tweet from a public dataset. Mentions and URLs are
stripped by the D10 normalisation. Examples naming third parties were skipped
where practical, and a small number of artist/band names remain because removing
them would destroy the example's point. Before anything ships publicly, decide
whether the report quotes these verbatim or paraphrases them — still open.

## 6. Known coverage gaps — NOT being fixed yet

**Artist / music distribution (creator-side).** Pilot item P007 asks how to upload
a child's music to Spotify. It is not `content_availability` (no existing content
is being requested) and not `product_feature_feedback` (the capability exists; it
is simply not this support channel).

**Current handling: `other_unclear`.** No `creator` or `artist_distribution`
intent is being added. If the pilot surfaces **multiple recurring examples** of
this domain, flag it for possible taxonomy revision *after* the pilot — never
mid-annotation. A single instance is not evidence of a class.

## 7. What would falsify this proposal

- ~~`social_praise` falling below 4%~~ — **resolved: removed in 0.2.0** (§8).
- `product_feature_feedback` falling below 4% → fold into the nearest technical
  intent. **Resolved for now:** 7 of 40 pilot labels, the second-largest class.
- `other_unclear` exceeding ~15% → taxonomy inadequate, revise before labelling.
- `billing ↔ plans` confusion above 25% *with* indistinguishable escalation
  outcomes → merge into one `billing_plans` intent.
- `playback ↔ app_device` confusion above 25% → **do not merge** (decision 2);
  instead sharpen BR-4 and re-pilot.

## 9. Pilot evidence behind 0.2.x

Forty dev-window items, **one annotator**, codebook 0.1.1, method in
`docs/pilot.md`. Reproduce:
`python scripts/annotate_pilot.py --validate outputs/pilot/pilot_items.csv`

**This pilot does not statistically validate the taxonomy.** Forty items, one
annotator, no second rater and no repeat pass: it is a design probe, not a
measurement. Every class count below carries an interval far wider than the count
itself — a class seen twice is consistent with anything from rare to common. The
three tiers below are kept separate so a reviewer can accept the observations
while rejecting the judgements.

### 9.1 Observed — what the annotation actually shows

| Observation | Value |
|---|---|
| Items annotated | 40 / 40 |
| `other_unclear` after lock | 7.5% (3 items) |
| Items marked `cannot_represent` | 0 |
| `confidence: low` used | 0 times |
| `notes` recorded | 0 items |
| Escalated items | 14 (35.0%) |
| Escalated items whose reason was account-family but whose intent was not `account_access` | 4 (P003, P008, P024, P029) |
| Classes with ≤1 item before re-annotation | 2 (`playback_playlist`, `social_praise`) |
| `social_praise` labels | 1 (P001) |
| Intents showing both ESCALATE and AUTO_OK | 3 |

### 9.2 Judgement — annotator and reviewer decisions on top of those observations

These are **arguable calls**, not findings.

- **`social_praise` removed.** Its single label, P001 *"love me"*, was judged not
  to be praise — two words, no addressee, no evaluation of the brand, most
  plausibly a song-title request retrieved because the praise seed matched the
  bare word "love". *Judgement:* zero valid instances. A different reader could
  call it praise and keep the class; the evidence does not settle it.
- **BR-11 added; BR-4 amended.** The four coherence breaks in §9.1 were judged to
  be intent errors caused by device/OS wording, for three of them. The rule
  already existed in `app_device_technical`'s exclusions; promoting it to a
  numbered rule reflects a judgement about how annotators read a codebook, not a
  measurement.
- **P029 judged NOT an intent error.** Same coherence flag, different cause: a
  reason-code mismatch. This is why the check is advisory.
- **Six re-annotations applied** (§9.3). Four are direct BR-11 applications; one
  is forced by the class removal; **P014 is a weak call** and is flagged as such.
- **BR-6 clarified** from P020/P037, two items. Two items are not a pattern; the
  clarification was adopted because the rule was genuinely underspecified, not
  because the data demanded it.

### 9.3 Approved re-annotations

| ID | before → after | basis |
|---|---|---|
| P001 | `social_praise` → `other_unclear` | forced by class removal |
| P003 | `app_device_technical` → `account_access` | BR-11 |
| P008 | `app_device_technical` → `account_access` | BR-11 |
| P024 | `app_device_technical` → `account_access` | BR-11 |
| P030 | `app_device_technical` → `account_access` | BR-11 |
| P014 | `app_device_technical` → `playback_playlist` | BR-4 amendment — **weak call** |
| P029 | *unchanged* (`plans_eligibility`) | reviewed; reason-code issue, not intent |

Resulting distribution: `billing_subscription` 8, `product_feature_feedback` 7,
`account_access` 6, `content_availability` 6, `app_device_technical` 5,
`other_unclear` 3, `plans_eligibility` 3, `playback_playlist` 2.

### 9.4 Remaining ambiguity — unresolved, carried into the golden set

- **P014** — `playback_playlist` vs `content_availability` turns on whether the
  track is in catalogue, which the text does not settle. BR-3 has no tiebreaker
  for "can't listen to X" with no other signal.
- **P006** — entitlement vs client defect. See §10.
- **The pilot's own diagnostics were not exercised.** No `confidence: low`, no
  notes, so the hesitation signal a pilot exists to capture was not captured. The
  revisions rest on label patterns and the coherence check instead. A second
  annotator, or a repeat pass by the same one, would test what this pilot cannot.
- **Data-handling note.** The annotation round-trip through Excel corrupted
  non-ASCII text in 10 of 40 rows (smart quotes, emoji, `å`/`ö`, Thai). All were
  restored by exact match against `twcs.csv` and verified 40/40. Any future
  labelling must avoid a plain Excel CSV save, or repeat this verification.

## 10. Known ambiguity retained without a new intent

**Entitlement vs client defect.** Pilot item P006 — *"no matter how many times i
hit the x it won't go away. i pay for premium... it takes up 1/4 of the screen."*
This reads either as an app UI defect (`app_device_technical`) or as an
entitlement failure — paying for premium and still being served ads
(`billing_subscription`). No boundary rule separates them, and **none is being
added**: one instance is not evidence of a class, and inventing a rule from a
single item is how codebooks accumulate dead weight.

Annotator guidance: label by whichever reading dominates the message and record
the alternative in `notes`. If the golden set shows a recurring population, add a
**rule**, not an intent.
