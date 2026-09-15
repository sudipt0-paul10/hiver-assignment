#!/usr/bin/env python3
"""Memory-bounded brand profiling for the TWCS customer-support dataset.

Reproduces every statistic quoted in ``docs/profile.md`` directly from the
official ``twcs/twcs.csv`` (~493 MB, 2.81 M rows) on a 3 GB machine.

The file is never loaded whole. Three bounded stages:

1. ``stream_structure``  streaming pass over 4 narrow columns, keeping only
   fixed-width numpy arrays (~30 MB) plus a per-brand boolean mask. Text is
   not read.
2. ``build_threads``     vectorised union-find (pointer jumping) over the
   reply edges to recover conversation threads. No Python-level loop over rows.
3. ``stream_text``       second streaming pass that materialises ``text`` and
   ``created_at`` only for tweets inside the selected brands' threads
   (~180 k rows, ~40 MB).

Peak RSS in practice is ~350 MB.

Usage
-----
    python scripts/profile.py --csv twcs/twcs.csv
    python scripts/profile.py --csv twcs/twcs.csv --brands SpotifyCares AmericanAir
    python scripts/profile.py --csv twcs/twcs.csv --leakage --json profile.json

``--leakage`` additionally runs the TF-IDF 1-NN retrieval probe, which needs
scikit-learn. Everything else depends only on pandas and numpy.

Determinism: this script performs no sampling and no model fitting beyond a
deterministic TF-IDF, so repeated runs on the same CSV produce identical output.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import resource
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Text patterns. Mirrors the normalisation contract in docs/decisions.md:
# signatures / URLs / mentions are stripped for MEASUREMENT and RETRIEVAL only.
# Raw text is always preserved for generation.
# --------------------------------------------------------------------------
URL = re.compile(r"https?://\S+")
MENTION = re.compile(r"@\w+")
# Agent sign-offs: "^JD", "/RS", "*DL", "- Ben", "~AB"
SIGNATURE = re.compile(r"(?:[\^~*\-/]\s?[A-Za-z]{1,3}|-\s?[A-Z][a-z]{2,10})\s*$")
DM_DEFLECT = re.compile(
    r"\b(?:dm|direct message|private message|pm us|secure(?:d)? link)\b", re.I
)

# Keyword proxies. These are PROXIES, not labels: they bound prevalence to
# decide feasibility, and are never used as training or evaluation targets.
MUST_PROXY = {
    "account_access": r"\b(?:can.?t (?:log|sign) ?in|locked out|hacked|account (?:is )?(?:locked|disabled|suspended|compromis)|password|reset my account)\b",
    "payment_dispute": r"\b(?:charged (?:me )?(?:twice|again)|double charg|unauthoriz|didn.?t authorize|refund|money back|fraudulent|wrong(?:ly)? charg|cancel my (?:subscription|premium)|still being charged)\b",
    "security_privacy": r"\b(?:hacked|phish|scam|fraud|stolen|breach|identity)\b",
    "legal_rights": r"\b(?:lawyer|legal|sue|lawsuit|attorney|gdpr|ombudsman|regulator)\b",
    "safety": r"\b(?:threat|assault|unsafe|emergency|medical|discriminat|racist|harass)\b",
}

# Broad topic families, evaluated over ALL openers (not just the test window).
# Reproduces the §3 "Openers - the classification task" table in docs/profile.md.
# Groups are non-capturing purely to silence pandas' match-group warning; the
# matching behaviour of str.contains is unchanged.
OPENER_TOPICS = {
    "money/refund": r"\b(?:refund|charge|charged|billing|bill|payment|paid|invoice|subscription|premium|money|price|overcharg)\b",
    "account/login": r"\b(?:account|login|log in|password|sign in|signin|username|locked|hacked|verify|verification)\b",
    "technical/bug": r"\b(?:not work|doesn.?t work|broken|bug|crash|error|glitch|won.?t (?:play|load|open)|freez|stuck|offline)\b",
    "anger/profanity": r"\b(?:fuck|shit|wtf|useless|garbage|terrible|awful|worst|disgust|ridiculous|furious|angry|pissed)\b",
    "praise": r"\b(?:thank you|thanks|thx|love you|awesome|amazing|great job|appreciate|best)\b",
    "delay/cancel": r"\b(?:delay|delayed|cancel|cancelled|canceled|rebook|missed (?:my )?(?:flight|connection)|stranded)\b",
    "baggage": r"\b(?:bag|bags|baggage|luggage|suitcase|carry.?on|checked bag)\b",
}

INTENT_PROXY = {
    "billing_subscription": r"\b(?:premium|subscription|billing|charged|payment|family plan|student)\b",
    "content_availability": r"\b(?:album|song|artist|podcast|episode|available|release|missing from)\b",
    "playback_technical": r"\b(?:won.?t play|not play|skip|offline|download|shuffle|crash|buffer|glitch)\b",
    "praise_social": r"\b(?:thank|thanks|love|awesome|amazing|appreciate|best)\b",
    "delay_cancel": r"\b(?:delay|cancel|rebook|missed (?:my )?(?:flight|connection)|stranded|diverted)\b",
    "baggage": r"\b(?:bag|baggage|luggage|suitcase|carry.?on)\b",
    "seat_booking": r"\b(?:seat|upgrade|boarding pass|check.?in|reservation|record locator|booking)\b",
    "loyalty": r"\b(?:miles|aadvantage|loyalty|status|elite|points)\b",
}

# Temporal split on OPENER date. Whole threads follow their opener, which makes
# boundary-straddling threads impossible by construction (verified: 0).
SPLITS = {
    "train": ("2017-10-01", "2017-11-06"),
    "dev": ("2017-11-06", "2017-11-20"),
    "test": ("2017-11-20", "2017-12-04"),
}

GOLDEN_CORE_SIZES = (160, 200)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def peak_mb() -> float:
    """Peak resident set size in MB (Linux reports ru_maxrss in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval. Inlined to avoid a scipy/statsmodels dependency."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def norm_for_match(s: str) -> str:
    """Mask URLs/mentions, lowercase. Keeps signatures (so <>strip_signature)."""
    s = URL.sub(" <URL> ", str(s))
    s = MENTION.sub(" <USER> ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def strip_signature(s: str) -> str:
    """Full evaluation normalisation: drop URLs, mentions and the agent sign-off."""
    s = URL.sub(" ", str(s))
    s = MENTION.sub(" ", s)
    s = SIGNATURE.sub("", s.strip())
    return re.sub(r"\s+", " ", s).strip().lower()


def pct(x: float) -> float:
    return round(float(x) * 100, 2)


# --------------------------------------------------------------------------
# stage 1 - streaming structural pass (no text)
# --------------------------------------------------------------------------
def stream_structure(csv: str, brands: list[str], chunksize: int) -> dict[str, Any]:
    tweet_id, inbound, parent = [], [], []
    masks: dict[str, list[np.ndarray]] = {b: [] for b in brands}
    outbound_counts: pd.Series | None = None
    rows = 0

    cols = ["tweet_id", "author_id", "inbound", "in_response_to_tweet_id"]
    for chunk in pd.read_csv(
        csv,
        usecols=cols,
        chunksize=chunksize,
        dtype={"tweet_id": "int64", "author_id": "str", "in_response_to_tweet_id": "float64"},
    ):
        rows += len(chunk)
        is_in = chunk["inbound"].astype(str).str.lower().eq("true").to_numpy()

        tweet_id.append(chunk["tweet_id"].to_numpy(np.int64))
        inbound.append(is_in)
        par = chunk["in_response_to_tweet_id"].to_numpy(np.float64)
        parent.append(np.where(np.isnan(par), -1, par).astype(np.int64))

        # .eq() compares inside pandas; materialising an object ndarray per chunk
        # just to compare it would allocate a needless 400k-element copy.
        for b in brands:
            masks[b].append(chunk["author_id"].eq(b).to_numpy() & ~is_in)

        counts = chunk.loc[~is_in, "author_id"].value_counts()
        outbound_counts = counts if outbound_counts is None else outbound_counts.add(counts, fill_value=0)

    return {
        "rows": rows,
        "tweet_id": np.concatenate(tweet_id),
        "inbound": np.concatenate(inbound),
        "parent": np.concatenate(parent),
        "masks": {b: np.concatenate(v) for b, v in masks.items()},
        "outbound_counts": outbound_counts.sort_values(ascending=False),
    }


# --------------------------------------------------------------------------
# stage 2 - vectorised union-find over reply edges
# --------------------------------------------------------------------------
def build_threads(tweet_id: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, dict]:
    order = np.argsort(tweet_id)
    sorted_ids = tweet_id[order]

    pidx = np.full(len(tweet_id), -1, dtype=np.int64)
    has_parent = parent >= 0
    vals = parent[has_parent]
    pos = np.clip(np.searchsorted(sorted_ids, vals), 0, len(sorted_ids) - 1)
    resolved = sorted_ids[pos] == vals
    tmp = np.full(len(vals), -1, dtype=np.int64)
    tmp[resolved] = order[pos[resolved]]
    pidx[has_parent] = tmp

    # pointer jumping until fixed point; threads are shallow so this converges fast
    root = np.where(pidx >= 0, pidx, np.arange(len(tweet_id)))
    jumps = 0
    for jumps in range(1, 64):
        nxt = root[root]
        if np.array_equal(nxt, root):
            break
        root = nxt

    stats = {
        "parent_refs": int(has_parent.sum()),
        "edges_resolved": int((pidx >= 0).sum()),
        "dangling_refs": int(has_parent.sum() - (pidx >= 0).sum()),
        "jumps": jumps,
        "threads_total": int(len(np.unique(root))),
    }
    return root, stats


# --------------------------------------------------------------------------
# stage 3 - streaming text pass, restricted to the brands' threads
# --------------------------------------------------------------------------
def stream_text(csv: str, needed: np.ndarray, chunksize: int) -> pd.DataFrame:
    """Materialise text/author/timestamp ONLY for the selected brands' threads.

    author_id is pulled here rather than in a separate full-file read: reading
    2.8 M author strings unchunked would cost ~200 MB for the ~180 k we need.
    """
    want = pd.Index(needed)
    parts = []
    for chunk in pd.read_csv(
        csv,
        usecols=["tweet_id", "author_id", "created_at", "text"],
        chunksize=chunksize,
        dtype={"tweet_id": "int64", "author_id": "str", "created_at": "str", "text": "str"},
    ):
        parts.append(chunk[chunk["tweet_id"].isin(want)])
    out = pd.concat(parts, ignore_index=True)
    out["created_at"] = pd.to_datetime(
        out["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
    )
    return out


# --------------------------------------------------------------------------
# per-brand metrics
# --------------------------------------------------------------------------
def profile_brand(brand: str, frame: pd.DataFrame, leakage: bool) -> dict[str, Any]:
    brand_rows = frame[(frame["author_id_is_brand"]) & (~frame["inbound"])]
    # openers_all = every customer-opened thread. openers = those with a brand
    # reply. Kept distinct because the §3 topic table is defined over ALL openers.
    openers_all = frame[frame["is_root"] & frame["inbound"]]
    openers = openers_all.copy()

    first_reply = brand_rows.sort_values("created_at").groupby("thread").first()
    openers["reply"] = openers["thread"].map(first_reply["text"])
    openers = openers.dropna(subset=["reply"])

    sizes = frame.groupby("thread").size()
    n_customer = frame[frame["inbound"]].groupby("thread").size()

    raw = brand_rows["text"]
    masked = raw.map(norm_for_match)
    stripped = raw.map(strip_signature)

    out: dict[str, Any] = {
        "brand": brand,
        "threads": int(frame["thread"].nunique()),
        "brand_tweets": int(len(brand_rows)),
        "customer_tweets": int(frame["inbound"].sum()),
        "structure": {
            "thread_size_median": float(sizes.median()),
            "thread_size_mean": round(float(sizes.mean()), 2),
            "thread_size_p90": float(sizes.quantile(0.90)),
            "exactly_two_pct": pct((sizes == 2).mean()),
            "root_is_customer_pct": pct(frame[frame["is_root"]]["inbound"].mean()),
            "customer_followup_pct": pct((n_customer > 1).mean()),
        },
        "reply_character": {
            "dup_url_masked_pct": pct(1 - masked.nunique() / len(masked)),
            "dup_signature_stripped_pct": pct(1 - stripped.nunique() / len(stripped)),
            "has_signature_pct": pct(
                raw.map(lambda s: bool(SIGNATURE.search(URL.sub("", str(s)).strip()))).mean()
            ),
            "has_link_pct": pct(raw.str.contains(URL).mean()),
            "dm_deflection_pct": pct(raw.str.contains(DM_DEFLECT).mean()),
            "median_chars": float(raw.str.len().median()),
        },
        "first_action": {
            "dm_pct": pct(first_reply["text"].str.contains(DM_DEFLECT).mean()),
            "link_pct": pct(first_reply["text"].str.contains(URL).mean()),
        },
        "openers": {
            "n": int(len(openers)),
            "n_all": int(len(openers_all)),
            "median_chars": float(openers["text"].str.len().median()),
            "has_link_pct": pct(openers["text"].str.contains(URL).mean()),
            "repeat_customer_pct": pct(1 - openers["author_id"].nunique() / len(openers)),
            "received_reply_pct": pct(len(openers) / len(openers_all)),
        },
        # docs/profile.md §3 topic table - over ALL openers, whole date range
        "opener_topics_all": {
            k: pct(openers_all["text"].str.contains(v, case=False, regex=True).mean())
            for k, v in OPENER_TOPICS.items()
        },
    }

    # docs/profile.md §3 reply-template table. Uses the D10 normalisation
    # (strip_signature), so templates differing only in sign-off or URL collapse.
    template_counts = stripped.value_counts()
    out["reply_templates"] = {
        "distinct": int(len(template_counts)),
        "top10_share_pct": pct(template_counts.head(10).sum() / len(stripped)),
        "top3": [{"count": int(c), "text": t} for t, c in template_counts.head(3).items()],
    }

    latency = (
        (first_reply["created_at"] - openers.set_index("thread")["created_at"].reindex(first_reply.index))
        .dt.total_seconds()
        .div(60)
        .dropna()
    )
    out["first_action"]["latency_median_min"] = round(float(latency.median()), 1)
    out["first_action"]["latency_p90_min"] = round(float(latency.quantile(0.90)), 1)

    # temporal split on opener date
    seg = {}
    for name, (lo, hi) in SPLITS.items():
        seg[name] = openers[(openers["created_at"] >= lo) & (openers["created_at"] < hi)]
    out["split"] = {k: int(len(v)) for k, v in seg.items()}
    out["dm_rate_drift"] = {
        k: round(float(v["reply"].str.contains(DM_DEFLECT).mean()), 3) for k, v in seg.items()
    }

    train, test = seg["train"], seg["test"]
    train_replies = set(train["reply"].map(strip_signature))
    test_replies = test["reply"].map(strip_signature)
    out["leakage"] = {
        "test_customers_in_train_pct": pct(
            len(set(test["author_id"]) & set(train["author_id"])) / max(test["author_id"].nunique(), 1)
        ),
        "threads_straddling_boundary": int(
            (frame[frame["thread"].isin(test["thread"])].groupby("thread")["created_at"].min()
             < pd.Timestamp(SPLITS["test"][0], tz="UTC")).sum()
        ),
        "test_reply_verbatim_in_train_pct": pct(test_replies.isin(train_replies).mean()),
    }

    # MUST prevalence proxy + resulting statistical power
    any_must = np.zeros(len(test), dtype=bool)
    by_family = {}
    for fam, pattern in MUST_PROXY.items():
        hit = test["text"].str.contains(pattern, case=False, regex=True).to_numpy()
        by_family[fam] = pct(hit.mean())
        any_must |= hit
    prevalence = float(any_must.mean())
    power = {}
    for core in GOLDEN_CORE_SIZES:
        n_must = int(round(core * prevalence))
        if n_must < 2:
            power[str(core)] = {"n_must": n_must, "verdict": "too few to measure"}
        else:
            lo, hi = wilson(int(round(n_must * 0.9)), n_must)
            power[str(core)] = {
                "n_must": n_must,
                "recall90_ci": [round(lo, 3), round(hi, 3)],
                "half_width_pp": round((hi - lo) / 2 * 100, 1),
            }
    out["must_proxy"] = {"families": by_family, "any_pct": pct(prevalence), "power": power}

    out["intent_proxy"] = {
        k: pct(test["text"].str.contains(v, case=False, regex=True).mean())
        for k, v in INTENT_PROXY.items()
    }

    if leakage:
        out["retrieval_probe"] = _retrieval_probe(train, test)
    return out


def _nn1_cosine(xte, xtr, block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 cosine neighbour of each test row among the train rows.

    TfidfVectorizer L2-normalises its output, so for unit rows
    ``cosine(a, b) == a @ b``. This is therefore numerically identical to
    ``NearestNeighbors(n_neighbors=1, metric="cosine")``, whose brute-force path
    materialises the entire |test| x |train| distance matrix — 6,808 x 14,547
    here, ~0.8 GB dense, which measured as the single largest allocation in the
    whole profile (peak RSS 760 MB -> 2,440 MB). Blocking bounds that to
    ``block x |train|`` floats at a time.
    """
    n = xte.shape[0]
    best_idx = np.empty(n, dtype=np.int64)
    best_sim = np.empty(n, dtype=np.float64)
    xtr_t = xtr.T.tocsc()
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = (xte[start:stop] @ xtr_t).toarray()
        loc = sims.argmax(axis=1)
        best_idx[start:stop] = loc
        best_sim[start:stop] = sims[np.arange(stop - start), loc]
        del sims
    return best_sim, best_idx


def _retrieval_probe(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """TF-IDF 1-NN over openers. Establishes an honest floor for how much a
    simple retriever recovers the historical reply. Requires scikit-learn."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return {"skipped": "scikit-learn not installed"}

    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=200_000, sublinear_tf=True)
    xtr = vec.fit_transform(train["text"].map(strip_signature))
    xte = vec.transform(test["text"].map(strip_signature))
    sim, idx = _nn1_cosine(xte, xtr)

    retrieved = train["reply"].map(strip_signature).to_numpy()[idx]
    gold = test["reply"].map(strip_signature).to_numpy()

    rv = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(
        np.concatenate([retrieved, gold])
    )
    a, b = rv.transform(retrieved), rv.transform(gold)
    num = np.asarray(a.multiply(b).sum(1)).ravel()
    den = np.sqrt(np.asarray(a.multiply(a).sum(1)).ravel() * np.asarray(b.multiply(b).sum(1)).ravel())
    reply_sim = num / (den + 1e-9)

    return {
        "opener_sim_median": round(float(np.median(sim)), 3),
        "opener_sim_ge_0.9_pct": pct((sim >= 0.9).mean()),
        "retrieved_reply_equals_gold_pct": pct((retrieved == gold).mean()),
        "retrieved_vs_gold_cosine_median": round(float(np.median(reply_sim)), 3),
    }


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="twcs/twcs.csv", help="path to the official twcs.csv")
    ap.add_argument("--brands", nargs="+", default=["SpotifyCares", "AmericanAir"])
    ap.add_argument("--chunksize", type=int, default=400_000)
    ap.add_argument("--top-brands", type=int, default=10)
    ap.add_argument("--leakage", action="store_true", help="also run the TF-IDF 1-NN probe (needs scikit-learn)")
    ap.add_argument("--json", help="write the full result to this path")
    args = ap.parse_args()

    t0 = time.time()
    struct = stream_structure(args.csv, args.brands, args.chunksize)
    root, tstats = build_threads(struct["tweet_id"], struct["parent"])

    report: dict[str, Any] = {
        "source_csv": args.csv,
        "total_rows": struct["rows"],
        "inbound_share": round(float(struct["inbound"].mean()), 3),
        "distinct_outbound_brands": int(len(struct["outbound_counts"])),
        "thread_graph": tstats,
        "top_brands": {k: int(v) for k, v in struct["outbound_counts"].head(args.top_brands).items()},
        "brands": {},
    }

    is_root = root == np.arange(len(root))
    needed_threads = {b: np.unique(root[struct["masks"][b]]) for b in args.brands}

    # Restrict to the selected brands' threads BEFORE building any DataFrame:
    # carrying all 2.81 M rows into base and then merging 179 k out of it wastes
    # ~56 MB and makes the join far more expensive than it needs to be.
    keep = np.zeros(len(root), dtype=bool)
    for threads in needed_threads.values():
        keep |= np.isin(root, threads)
    needed_ids = struct["tweet_id"][keep]

    text = stream_text(args.csv, needed_ids, args.chunksize)

    base = pd.DataFrame(
        {
            "tweet_id": struct["tweet_id"][keep],
            "inbound": struct["inbound"][keep],
            "thread": root[keep],
            "is_root": is_root[keep],
        }
    )
    for b in args.brands:
        base[f"__{b}"] = struct["masks"][b][keep]

    merged = base.merge(text, on="tweet_id", how="inner")
    # the 2.81 M-row structural arrays are dead once `merged` exists
    del struct, root, is_root, keep, base, text
    gc.collect()

    for b in args.brands:
        sub = merged[merged["thread"].isin(needed_threads[b])].copy()
        sub["author_id_is_brand"] = sub[f"__{b}"]
        report["brands"][b] = profile_brand(b, sub, args.leakage)

    report["runtime_sec"] = round(time.time() - t0, 1)
    report["peak_rss_mb"] = round(peak_mb(), 1)

    print(json.dumps(report, indent=2, default=str))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\n[written] {args.json}", file=sys.stderr)
    print(
        f"\n[profile] {report['total_rows']:,} rows | {report['runtime_sec']}s | "
        f"peak RSS {report['peak_rss_mb']} MB",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
