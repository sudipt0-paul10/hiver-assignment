# Prior Research & Dataset Evidence

**STATUS: PRIOR EVIDENCE — REPRODUCE IMPORTANT NUMBERS BEFORE FINAL REPORT**

## SpotifyCares profile from prior extract
- 43,265 brand tweets
- 48,543 customer tweets
- 28,280 threads
- 28,221 openers
- 41.9% customer follow-ups
- median thread size 2; p90 6
- 62.9% of threads had exactly 2 tweets
- 15.7% of openers had links
- 4.5% likely non-English
- 13.2% money/refund/charge pattern
- 20.6% account/booking-ID-like pattern
- 5.7% playback-bug pattern
- 1.9% praise
- 4.3% profanity/anger
- 24.5% exact normalized duplicate brand replies
- 97.1% brand replies contained signatures
- 50.5% brand replies contained links
- 92.5% openers received a direct brand reply
- first action: ~37.2% private/DM, ~23.6% information/link, ~19.3% clarification
- median first reply latency ~62.3 minutes
- ~99.5% dated Oct 1–Dec 3 2017
- ~12.7% openers from repeat customers

## AmericanAir comparison
Prior extract:
- 36,764 brand tweets
- 50,054 customer tweets
- 26,386 threads
- 47.8% customer follow-ups
- 16.9% delay/cancel
- 9.5% baggage
- 9.1% anger
- 17.4% first actions private
- 20.5% social acknowledgement
- 17.2% apology/empathy
This supported keeping AmericanAir as a runner-up.

## Provenance warning
The earlier profiling extracts were obtained because Kaggle/Hugging Face were unreachable from the analysis environment. They were used for exploration only. Re-derive important statistics from the official `twcs.csv` before implementation/final reporting.

## Leakage evidence from prior Spotify probe
Previous temporal split:
- train: Oct 1–Nov 5
- dev: Nov 6–Nov 19
- test: Nov 20–Dec 3

Observed:
- ~3.4% of test customers appeared in train
- 288 threads straddled the test boundary
- ~1.2% test items had near-duplicate cosine >=0.9
- ~21.4% of test reference replies appeared verbatim in train replies
- historical DM rate train/dev/test was ~0.37/~0.314/~0.432

These observations motivate explicit leakage controls.

## Model probes
Prior inexpensive probes:
- TF-IDF + Logistic Regression predicting a historical DM proxy: AUC ~0.889 on the full temporal probe;
- smaller samples: ~0.811 at 100, ~0.852 at 400, ~0.871 at 1,600;
- keyword silver-label probe was directional, not ground truth;
- 1-NN action agreement ~0.46 vs majority ~0.436.

These are exploratory/proxy results, not final performance.

## Response/retrieval observations
Repeated historical brand replies make retrieval promising, but also make leakage particularly dangerous. Explicitly test self-retrieval, same-thread retrieval, near-duplicate retrieval, customer overlap, and temporal leakage.

## External research themes
Prior research suggested:
- retrieval + generation is a common support-agent pattern;
- BLEU/ROUGE are weak proxies for human support quality;
- LLM judges can have bias/drift and should be validated against humans;
- selective prediction / risk–coverage is appropriate when agents can abstain/escalate;
- support datasets often contain repeated templates, making leakage checks essential.

These themes are methodological guidance, not evidence of final system quality.
