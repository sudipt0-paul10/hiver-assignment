#!/usr/bin/env python3
"""Unsupervised intent-taxonomy discovery from TRAIN-split openers only.

Feeds decision D17 in ``docs/decisions.md``. Produces *candidate* classes and the
evidence needed to judge them. It does not lock a taxonomy, label anything, or
call any API.

Leakage discipline
------------------
Reads **only** the train window (see ``profile.SPLITS``). Dev is left free for
validating the taxonomy later; test is never opened. The split is on opener date
with whole threads following their opener, so no test conversation can leak in.

Method
------
1. TF-IDF over D10-normalised openers (signatures/URLs/mentions stripped).
2. NMF for a range of k -- chosen over KMeans as the primary lens because soft,
   parts-based topics read more honestly on short support text; KMeans
   silhouette is reported alongside as a structure sanity check.
3. For every candidate class, three pieces of evidence:
   - **prevalence**, against the >=4% viability floor from D17;
   - **historical action profile** (DM-deflection rate, link rate of the first
     brand reply) -- D17 says split only where the distinction changes what the
     system does, so classes whose action profiles coincide are merge candidates;
   - **MUST-proxy mass**, reusing ``profile.MUST_PROXY``, showing which classes
     carry escalation weight.
4. A separate check of whether conversational state (request / social / unclear)
   behaves as its own dimension rather than as sibling topic labels.

Everything is seeded; repeated runs on the same CSV give identical output.

Usage
-----
    python scripts/taxonomy.py --csv twcs/twcs.csv
    python scripts/taxonomy.py --csv twcs/twcs.csv --k 8 --json outputs/taxonomy.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile as P  # noqa: E402  (same directory; reuses the audited loaders)

warnings.filterwarnings("ignore")
SEED = 20260912

# Conversational state, tested as a possible SECOND dimension rather than as
# topic classes (D17 flags the previous flat taxonomy for mixing the two).
STATE = {
    "social_nonrequest": r"\b(?:thank|thanks|thx|love (?:you|u)|awesome|amazing|appreciate|happy birthday|congrats)\b",
    "dm_followup": r"\b(?:dm(?:'?d| sent| you)|sent (?:you )?a dm|replied to your dm|in your dms|check your dm)\b",
    "explicit_question": r"\?",
}


# Candidate-intent seed patterns. These exist ONLY to (a) retrieve real training
# examples for human curation and (b) produce a rough prevalence estimate. They
# are NOT the codebook rules and they are NOT labels — docs/taxonomy.md defines
# the intents in prose, and a human applies them. Applied in the listed order to
# force mutual exclusivity for the estimate; that ordering is a crude device, not
# a decision procedure.
CANDIDATE_SEEDS = {
    "account_access": r"\b(?:can.?t (?:log|sign) ?in|cant (?:log|sign) ?in|log ?in|login|sign ?in|password|hacked|locked out|account (?:is )?(?:locked|disabled|suspended|hijack|compromis)|someone (?:else )?(?:is us|has|used) my|recover my account|verify my account)\b",
    "billing_subscription": r"\b(?:charged|charge|charging|refund|money back|payment (?:failed|declined|issue)|billed|billing|invoice|double charg|cancel (?:my )?(?:premium|subscription)|unsubscrib|still (?:being )?charg)\b",
    "plans_eligibility": r"\b(?:family plan|premium family|student (?:discount|plan|verif)|duo|upgrade to premium|free trial|promo|offer|discount|eligib|how much (?:is|does)|price|switch (?:to|my) plan|join (?:the )?family)\b",
    "content_availability": r"\b(?:not available|isn.?t available|unavailable|available in|not (?:on|in) spotify|missing from|removed|take[n]? off|why (?:isn.?t|is|aren.?t|are).{0,30}(?:on spotify|available)|release|new album|when will .{0,20}be (?:on|available))\b",
    "app_device_technical": r"\b(?:app (?:crash|keeps|won.?t|is|not)|crash|crashing|update the app|latest version|ios|android|apple watch|windows|web player|desktop app|chromecast|alexa|google home|car ?play|bluetooth|battery|connect to)\b",
    "playback_playlist": r"\b(?:playlist|shuffle|skip|queue|offline|download(?:ed|ing)?|repeat|won.?t play|not play|stops? playing|keeps? pausing|song(?:s)? play|listen)\b",
    "social_praise": r"\b(?:thank you|thanks|thx|love (?:you|u|spotify)|you.?re the best|amazing|awesome|appreciate|happy birthday|congrats)\b",
}
SEED_ORDER = [
    "account_access",
    "billing_subscription",
    "plans_eligibility",
    "content_availability",
    "app_device_technical",
    "playback_playlist",
    "social_praise",
]


def seed_assign(df: pd.DataFrame) -> pd.Series:
    """First-match-wins assignment over CANDIDATE_SEEDS; unmatched -> other_unclear."""
    out = pd.Series("other_unclear", index=df.index, dtype=object)
    unset = pd.Series(True, index=df.index)
    for name in SEED_ORDER:
        hit = df["text"].str.contains(CANDIDATE_SEEDS[name], case=False, regex=True) & unset
        out[hit] = name
        unset &= ~hit
    return out


def dump_examples(df: pd.DataFrame, per_intent: int) -> dict[str, Any]:
    """Deterministic real-example pool per candidate intent, for human curation."""
    df = df.copy()
    df["seed_intent"] = seed_assign(df)
    rng = np.random.default_rng(SEED)
    pool: dict[str, Any] = {}
    for name in SEED_ORDER + ["other_unclear"]:
        sub = df[df["seed_intent"] == name]
        if len(sub) == 0:
            pool[name] = {"n": 0, "examples": []}
            continue
        take = rng.choice(len(sub), size=min(per_intent, len(sub)), replace=False)
        rows = sub.iloc[np.sort(take)]
        must = np.zeros(len(sub), dtype=bool)
        for pat in P.MUST_PROXY.values():
            must |= sub["text"].str.contains(pat, case=False, regex=True).to_numpy()
        pool[name] = {
            "n": int(len(sub)),
            "prevalence_pct": P.pct(len(sub) / len(df)),
            "reply_dm_pct": P.pct(sub["reply_is_dm"].mean()),
            "reply_link_pct": P.pct(sub["reply_has_link"].mean()),
            "must_proxy_pct": P.pct(must.mean()),
            "examples": [
                {"text": re.sub(r"\s+", " ", r["clean"])[:180], "dm": bool(r["reply_is_dm"])}
                for _, r in rows.iterrows()
            ],
        }
    return pool


def load_train_openers(csv: str, brand: str, chunksize: int) -> pd.DataFrame:
    """Reconstruct threads, then return this brand's TRAIN-window openers."""
    struct = P.stream_structure(csv, [brand], chunksize)
    root, _ = P.build_threads(struct["tweet_id"], struct["parent"])
    is_root = root == np.arange(len(root))

    threads = np.unique(root[struct["masks"][brand]])
    keep = np.isin(root, threads)
    ids = struct["tweet_id"][keep]
    text = P.stream_text(csv, ids, chunksize)

    base = pd.DataFrame(
        {
            "tweet_id": struct["tweet_id"][keep],
            "inbound": struct["inbound"][keep],
            "thread": root[keep],
            "is_root": is_root[keep],
            "is_brand": struct["masks"][brand][keep],
        }
    )
    frame = base.merge(text, on="tweet_id", how="inner")
    del struct, root, is_root, keep, base, text

    brand_rows = frame[frame["is_brand"] & ~frame["inbound"]]
    first = brand_rows.sort_values("created_at").groupby("thread").first()
    openers = frame[frame["is_root"] & frame["inbound"]].copy()
    openers["reply"] = openers["thread"].map(first["text"])
    openers = openers.dropna(subset=["reply"])

    lo, hi = P.SPLITS["train"]
    train = openers[(openers["created_at"] >= lo) & (openers["created_at"] < hi)].copy()
    train["clean"] = train["text"].map(P.strip_signature)
    train["reply_is_dm"] = train["reply"].str.contains(P.DM_DEFLECT)
    train["reply_has_link"] = train["reply"].str.contains(P.URL)
    return train.reset_index(drop=True)


def structure_scan(X, ks: list[int]) -> list[dict[str, Any]]:
    """KMeans silhouette across k: is there cluster structure at all?"""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(SEED)
    sample = rng.choice(X.shape[0], size=min(4000, X.shape[0]), replace=False)
    out = []
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=5, batch_size=1024)
        labels = km.fit_predict(X)
        out.append(
            {
                "k": k,
                "silhouette": round(float(silhouette_score(X[sample], labels[sample], metric="cosine")), 4),
                "inertia": round(float(km.inertia_), 1),
            }
        )
    return out


def nmf_topics(X, terms: np.ndarray, k: int, n_terms: int = 12):
    from sklearn.decomposition import NMF

    model = NMF(n_components=k, random_state=SEED, init="nndsvda", max_iter=400)
    W = model.fit_transform(X)
    top = [[terms[i] for i in comp.argsort()[::-1][:n_terms]] for comp in model.components_]
    return W, top, float(model.reconstruction_err_)


def describe(df: pd.DataFrame, assign: np.ndarray, k: int, top_terms, n_ex: int = 3):
    rows = []
    overall_dm = float(df["reply_is_dm"].mean())
    for c in range(k):
        m = assign == c
        sub = df[m]
        if len(sub) == 0:
            continue
        must = np.zeros(len(sub), dtype=bool)
        per_family = {}
        for fam, pat in P.MUST_PROXY.items():
            hit = sub["text"].str.contains(pat, case=False, regex=True).to_numpy()
            per_family[fam] = P.pct(hit.mean())
            must |= hit
        ex = [re.sub(r"\s+", " ", t)[:96] for t in sub["clean"].head(n_ex)]
        rows.append(
            {
                "cluster": c,
                "n": int(m.sum()),
                "prevalence_pct": P.pct(m.mean()),
                "viable_4pct": bool(m.mean() >= 0.04),
                "top_terms": top_terms[c],
                "reply_dm_pct": P.pct(sub["reply_is_dm"].mean()),
                "reply_dm_lift": round(float(sub["reply_is_dm"].mean() / overall_dm), 2),
                "reply_link_pct": P.pct(sub["reply_has_link"].mean()),
                "must_proxy_pct": P.pct(must.mean()),
                "must_families": {k2: v for k2, v in per_family.items() if v > 0},
                "examples": ex,
            }
        )
    return rows


def merge_candidates(rows: list[dict], dm_tol: float = 4.0, link_tol: float = 6.0):
    """Pairs whose historical action profiles are within tolerance of each other.

    D17: split only where the distinction changes what the system does. Two
    classes the brand treated identically are candidates to merge unless a human
    policy reason keeps them apart.
    """
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            d_dm = abs(a["reply_dm_pct"] - b["reply_dm_pct"])
            d_lk = abs(a["reply_link_pct"] - b["reply_link_pct"])
            if d_dm <= dm_tol and d_lk <= link_tol:
                out.append(
                    {
                        "pair": [a["cluster"], b["cluster"]],
                        "dm_gap_pp": round(d_dm, 2),
                        "link_gap_pp": round(d_lk, 2),
                        "a_terms": a["top_terms"][:5],
                        "b_terms": b["top_terms"][:5],
                    }
                )
    return out


def state_dimension(df: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for name, pat in STATE.items():
        hit = df["text"].str.contains(pat, case=False, regex=True)
        out[name] = {
            "prevalence_pct": P.pct(hit.mean()),
            "reply_dm_pct": P.pct(df.loc[hit, "reply_is_dm"].mean()) if hit.any() else None,
            "reply_dm_pct_when_absent": P.pct(df.loc[~hit, "reply_is_dm"].mean()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="twcs/twcs.csv")
    ap.add_argument("--brand", default="SpotifyCares")
    ap.add_argument("--chunksize", type=int, default=400_000)
    ap.add_argument("--k", type=int, default=8, help="components for the reported taxonomy")
    ap.add_argument("--scan", nargs="+", type=int, default=[4, 6, 8, 10, 12, 14])
    ap.add_argument("--dump-examples", type=int, metavar="N",
                    help="emit N real train examples per candidate intent for codebook curation, then exit")
    ap.add_argument("--json")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer

    df = load_train_openers(args.csv, args.brand, args.chunksize)
    lo, hi = P.SPLITS["train"]
    print(f"[train-only] {args.brand}: {len(df):,} openers in [{lo}, {hi})", file=sys.stderr)
    print(f"[train-only] date range actually used: {df.created_at.min()} .. {df.created_at.max()}", file=sys.stderr)

    if args.dump_examples:
        pool = dump_examples(df, args.dump_examples)
        out = {"brand": args.brand, "split_used": "train", "n_openers": int(len(df)), "pool": pool}
        print(json.dumps(out, indent=2, default=str))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, default=str)
            print(f"\n[written] {args.json}", file=sys.stderr)
        return 0

    vec = TfidfVectorizer(
        min_df=5, max_df=0.4, ngram_range=(1, 2), sublinear_tf=True, stop_words="english", max_features=60_000
    )
    X = vec.fit_transform(df["clean"])
    terms = np.array(vec.get_feature_names_out())

    scan = structure_scan(X, args.scan)
    W, top_terms, err = nmf_topics(X, terms, args.k)
    assign = W.argmax(axis=1)
    # rows with no signal at all shouldn't be silently forced into topic 0
    assign = np.where(W.max(axis=1) <= 0, -1, assign)

    rows = describe(df, assign, args.k, top_terms)
    report = {
        "brand": args.brand,
        "split_used": "train",
        "train_window": list(P.SPLITS["train"]),
        "n_openers": int(len(df)),
        "vocab": int(len(terms)),
        "unassigned_pct": P.pct((assign == -1).mean()),
        "kmeans_structure_scan": scan,
        "nmf_k": args.k,
        "nmf_reconstruction_err": round(err, 4),
        "clusters": rows,
        "merge_candidates": merge_candidates(rows),
        "state_dimension": state_dimension(df),
        "overall_reply_dm_pct": P.pct(df["reply_is_dm"].mean()),
    }
    print(json.dumps(report, indent=2, default=str))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\n[written] {args.json}", file=sys.stderr)
    print(f"[peak RSS] {P.peak_mb():.1f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
