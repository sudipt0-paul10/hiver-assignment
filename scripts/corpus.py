#!/usr/bin/env python3
"""Shared data layer for the D8 pipeline: threads, splits, opener/reply pairs.

    python scripts/corpus.py --build        # stream twcs.csv once, cache the brand's threads
    python scripts/corpus.py --summary      # split sizes and sanity checks

Every downstream stage imports from here instead of re-streaming the 493 MB CSV.
`scripts/profile.py` remains the single definition of thread reconstruction,
normalisation (D10) and the temporal split (D18); this module only assembles
those into the frames the pipeline needs, and caches the result.

The cache is a pickle under `cache/corpus/` (git-ignored). It holds only rows
already derivable from twcs.csv, so deleting it costs ~25 s, never correctness.
`--build` records the source file's SHA-256 and size; a stale cache is detected
and rebuilt rather than silently used.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile as P  # noqa: E402

CACHE_DIR = os.path.join("cache", "corpus")
BRAND = "SpotifyCares"


def _source_fingerprint(csv_path: str) -> dict[str, Any]:
    """Cheap but honest staleness key: size plus a hash of the first and last 8 MB.

    Hashing 493 MB on every import would cost more than rebuilding the cache.
    """
    size = os.path.getsize(csv_path)
    h = hashlib.sha256()
    with open(csv_path, "rb") as fh:
        h.update(fh.read(8 << 20))
        if size > (16 << 20):
            fh.seek(-(8 << 20), os.SEEK_END)
            h.update(fh.read())
    return {"size": size, "head_tail_sha256": h.hexdigest()}


def build(csv_path: str, brand: str, chunksize: int) -> pd.DataFrame:
    struct = P.stream_structure(csv_path, [brand], chunksize)
    root, _ = P.build_threads(struct["tweet_id"], struct["parent"])
    is_root = root == np.arange(len(root))
    threads = np.unique(root[struct["masks"][brand]])
    keep = np.isin(root, threads)
    text = P.stream_text(csv_path, struct["tweet_id"][keep], chunksize)
    frame = pd.DataFrame({
        "tweet_id": struct["tweet_id"][keep],
        "inbound": struct["inbound"][keep],
        "thread": root[keep],
        "is_root": is_root[keep],
    }).merge(text, on="tweet_id", how="inner")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"{brand}.pkl"), "wb") as fh:
        pickle.dump({"frame": frame, "brand": brand,
                     "source": _source_fingerprint(csv_path)}, fh, protocol=4)
    return frame


def load(csv_path: str = "twcs/twcs.csv", brand: str = BRAND,
         chunksize: int = 400_000) -> pd.DataFrame:
    """Cached thread frame for one brand. Rebuilds if the cache is absent or stale."""
    path = os.path.join(CACHE_DIR, f"{brand}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if blob.get("source") == _source_fingerprint(csv_path):
            return blob["frame"]
        print(f"[corpus] cache stale for {csv_path}; rebuilding", file=sys.stderr)
    return build(csv_path, brand, chunksize)


def opener_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per thread: the customer opener plus the brand's FIRST reply.

    `reply` is NaN when the thread got no brand reply. The golden set was drawn
    only from threads that did have one (docs/golden_set.md 1), but the column is
    kept nullable so the retrieval corpus and the population frame can differ.
    """
    brand_rows = frame[~frame["inbound"]]
    first_reply = (brand_rows.sort_values("created_at")
                   .groupby("thread")
                   .first()[["text", "tweet_id"]]
                   .rename(columns={"text": "reply", "tweet_id": "reply_tweet_id"}))
    op = frame[frame["is_root"] & frame["inbound"]].copy()
    op = op.join(first_reply, on="thread")
    op["split"] = op["created_at"].map(split_of)
    op["clean"] = op["text"].map(P.strip_signature)
    op["reply_clean"] = op["reply"].map(lambda s: P.strip_signature(s) if isinstance(s, str) else s)
    return op


def split_of(ts: pd.Timestamp) -> str:
    """Temporal split on the OPENER date (D18). Whole threads follow their opener."""
    for name, (lo, hi) in P.SPLITS.items():
        if pd.Timestamp(lo, tz="UTC") <= ts < pd.Timestamp(hi, tz="UTC"):
            return name
    return "outside"


def retrieval_corpus(pairs: pd.DataFrame) -> pd.DataFrame:
    """TRAIN openers that actually have a reply - the only evidence the agent may cite.

    Restricting to train is the primary leakage control (D18): a test message
    cannot retrieve its own thread or its own reply because neither is in the
    index. The remaining controls - customer-disjoint and near-duplicate
    suppression - are applied per query in scripts/retrieval.py, because they
    depend on which item is being answered.
    """
    c = pairs[(pairs["split"] == "train") & pairs["reply"].notna()].copy()
    return c[c["clean"].str.len() > 0].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--csv", default="twcs/twcs.csv")
    ap.add_argument("--brand", default=BRAND)
    ap.add_argument("--chunksize", type=int, default=400_000)
    args = ap.parse_args()

    frame = build(args.csv, args.brand, args.chunksize) if args.build else load(
        args.csv, args.brand, args.chunksize)
    if not (args.build or args.summary):
        ap.error("pass --build or --summary")

    pairs = opener_pairs(frame)
    print(f"brand           {args.brand}")
    print(f"tweets in threads {len(frame):,}   threads {frame['thread'].nunique():,}")
    print(f"openers          {len(pairs):,}   with a brand reply {int(pairs['reply'].notna().sum()):,}")
    print("\nopeners by split (thread follows its opener date, D18):")
    for s in ("train", "dev", "test", "outside"):
        sub = pairs[pairs["split"] == s]
        if len(sub):
            print(f"  {s:<8} {len(sub):>7,}   with reply {int(sub['reply'].notna().sum()):>7,}"
                  f"   {P.SPLITS.get(s, ('', ''))[0]} -> {P.SPLITS.get(s, ('', ''))[1]}")
    straddle = pairs.groupby("thread")["split"].nunique().gt(1).sum()
    print(f"\nthreads straddling a split boundary: {straddle}  (must be 0 by construction)")
    print(f"retrieval corpus (train openers with a reply): {len(retrieval_corpus(pairs)):,}")
    print(f"[peak RSS] {P.peak_mb():.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
