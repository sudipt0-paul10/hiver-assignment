#!/usr/bin/env python3
"""D8 retrieval: similar historical cases from TRAIN only, with the leakage controls.

    python scripts/retrieval.py --selftest      # prove the guards fire, on the golden set
    python scripts/retrieval.py --demo "my account got hacked"

Per D8 the retriever starts at TF-IDF; D15 forbids embeddings or a vector store
until an ablation on dev shows they earn their place, and D13 already records
that TF-IDF 1-NN is a weak retriever (retrieved reply equals the gold reply 0.4%
of the time). That is the reason reply quality is judged (D11) rather than scored
by overlap - not a reason to skip the measurement.

LEAKAGE CONTROLS (D18), and where each one lives
------------------------------------------------
1. **No self-retrieval, no same-thread retrieval** - structural. The index holds
   TRAIN openers only and every evaluated item is from TEST, so an item's own
   tweet and its own thread are not in the index at all. `exclude_threads` exists
   for the dev-tuning case, where query and index could share a window.
2. **Customer-disjoint** - per query. 7 of the 200 golden customers also appear
   in train (`customer_also_in_train` in golden_key.json). Their earlier threads
   are in the index, and letting an item retrieve its own customer's history is
   exactly the leak D18 names. `search(exclude_customers=...)` removes them.
3. **Near-duplicate suppression** - per query. A train opener that is a near
   copy of the query would hand over an almost-verbatim answer. The threshold and
   the shingle definition are imported from the golden sampler so retrieval and
   sampling cannot drift apart.

The guards are opt-in parameters, and `--selftest` reports how often each one
actually fires, so "we controlled for leakage" is a measurement here rather than
a claim.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C  # noqa: E402
import profile as P  # noqa: E402
import sample_golden as SG  # noqa: E402  (canonical near-duplicate definition)


class Retriever:
    """TF-IDF cosine over D10-normalised train openers, replies carried alongside."""

    def __init__(self, corpus_df: pd.DataFrame, ngram: tuple[int, int] = (1, 2),
                 min_df: int = 2, sublinear_tf: bool = True) -> None:
        self.df = corpus_df.reset_index(drop=True)
        self.vec = TfidfVectorizer(ngram_range=ngram, min_df=min_df,
                                   sublinear_tf=sublinear_tf, strip_accents="unicode")
        self.matrix = self.vec.fit_transform(self.df["clean"])
        self._shingles: list[frozenset[str]] | None = None

    @property
    def shingles(self) -> list[frozenset[str]]:
        if self._shingles is None:
            self._shingles = [SG.shingles(t) for t in self.df["clean"]]
        return self._shingles

    def search(self, text: str, k: int = 5, *,
               exclude_customers: Iterable[str] = (),
               exclude_threads: Iterable[int] = (),
               exclude_tweets: Iterable[int] = (),
               drop_near_dups: bool = True,
               oversample: int = 40) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Top-k evidence for one message, plus a count of what each guard removed."""
        q_clean = P.strip_signature(text)
        sims = (self.vec.transform([q_clean]) @ self.matrix.T).toarray().ravel()
        # oversample before filtering: the guards can remove the whole head of the ranking
        order = np.argsort(-sims)[:max(oversample, k * 8)]
        cust, thr, twt = set(exclude_customers), set(exclude_threads), set(exclude_tweets)
        q_sh = SG.shingles(q_clean) if drop_near_dups else None
        blocked = {"same_customer": 0, "same_thread": 0, "self": 0, "near_duplicate": 0}
        hits: list[dict[str, Any]] = []
        for i in order:
            r = self.df.iloc[int(i)]
            if int(r["tweet_id"]) in twt:
                blocked["self"] += 1
                continue
            if str(r["author_id"]) in cust:
                blocked["same_customer"] += 1
                continue
            if int(r["thread"]) in thr:
                blocked["same_thread"] += 1
                continue
            if q_sh is not None and SG._jaccard(q_sh, self.shingles[int(i)]) >= SG.NEAR_DUP_JACCARD:
                blocked["near_duplicate"] += 1
                continue
            hits.append({"tweet_id": int(r["tweet_id"]), "thread": int(r["thread"]),
                         "customer_id": str(r["author_id"]), "score": float(sims[i]),
                         "opener": r["text"], "reply": r["reply"]})
            if len(hits) == k:
                break
        return hits, blocked


def golden_guards(keyfile: str = "data/golden/golden_key.json") -> dict[str, dict[str, Any]]:
    """Per-golden-item retrieval guards, read straight off the sampling key."""
    import json
    key = json.load(open(keyfile, encoding="utf-8"))["items"]
    return {gid: {"customer_id": v["customer_id"], "thread_id": v["thread_id"],
                  "tweet_id": v["tweet_id"], "in_train": v["customer_also_in_train"]}
            for gid, v in key.items()}


def selftest(csv_path: str, k: int) -> int:
    import csv as _csv
    pairs = C.opener_pairs(C.load(csv_path))
    corp = C.retrieval_corpus(pairs)
    r = Retriever(corp)
    print(f"index: {len(corp):,} train openers, {r.matrix.shape[1]:,} features")

    guards = golden_guards()
    with open("data/golden/golden_annotated.csv", encoding="utf-8-sig", newline="") as fh:
        items = list(_csv.DictReader(fh))

    train_customers = set(corp["author_id"])
    print(f"\ngolden customers also present in the train index: "
          f"{sum(1 for g in guards.values() if g['customer_id'] in train_customers)} "
          f"(key says {sum(1 for g in guards.values() if g['in_train'])})")

    tot = {"same_customer": 0, "same_thread": 0, "self": 0, "near_duplicate": 0}
    top1, empties, leaked_unguarded = [], 0, []
    for it in items:
        g = guards[it["id"]]
        hits, blocked = r.search(it["text"], k=k,
                                 exclude_customers=[g["customer_id"]],
                                 exclude_threads=[g["thread_id"]],
                                 exclude_tweets=[g["tweet_id"]])
        for key_ in tot:
            tot[key_] += blocked[key_]
        if hits:
            top1.append(hits[0]["score"])
        else:
            empties += 1
        # what would have leaked with the guard off?
        raw, _ = r.search(it["text"], k=k, drop_near_dups=False)
        same = [h for h in raw if h["customer_id"] == g["customer_id"]]
        if same:
            leaked_unguarded.append((it["id"], g["customer_id"], len(same)))

    print("\n=== guard activations across all 200 golden items ===")
    for key_, n in tot.items():
        print(f"  {key_:<16} {n:>5} candidate(s) suppressed")
    print(f"\n  items left with no evidence after filtering: {empties}")
    s = pd.Series(top1)
    print(f"  top-1 cosine: median {s.median():.3f}  p10 {s.quantile(.10):.3f}  "
          f"p90 {s.quantile(.90):.3f}  min {s.min():.3f}")
    print("  (D13 measured median opener-to-opener similarity 0.303 on this data; a"
          " weak retriever is the expected starting point, not a bug)")

    print("\n=== customer-disjointness: what the guard actually prevented ===")
    if leaked_unguarded:
        for gid, cust, n in leaked_unguarded:
            print(f"  {gid}: {n} same-customer train opener(s) would have been retrieved "
                  f"(customer {cust})")
        print(f"  {len(leaked_unguarded)} item(s) affected - all blocked by exclude_customers.")
    else:
        print("  No same-customer candidate reached the top-k for any item even unguarded.")
        print("  The guard is still required: it must not depend on TF-IDF happening to rank")
        print("  the customer's own history low.")
    return 0


def demo(csv_path: str, text: str, k: int) -> int:
    pairs = C.opener_pairs(C.load(csv_path))
    r = Retriever(C.retrieval_corpus(pairs))
    hits, blocked = r.search(text, k=k)
    print(f"query: {text}\nblocked: {blocked}\n")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] score {h['score']:.3f}  thread {h['thread']}")
        print(f"    opener: {h['opener'][:150]}")
        print(f"    reply : {h['reply'][:150]}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", metavar="TEXT")
    ap.add_argument("--csv", default="twcs/twcs.csv")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    if args.demo:
        return demo(args.csv, args.demo, args.k)
    if args.selftest:
        return selftest(args.csv, args.k)
    ap.error("pass --selftest or --demo TEXT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
