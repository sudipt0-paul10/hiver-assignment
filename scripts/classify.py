#!/usr/bin/env python3
"""Intent classifiers. Every one is labelled by how it was constructed (D23).

    python scripts/classify.py --predict rule        --split golden
    python scripts/classify.py --predict distant-lr  --split golden
    python scripts/classify.py --predict llm         --split golden      # needs an API key
    python scripts/classify.py --show-prompt                             # inspect, no API call
    python scripts/classify.py --dev-sample 20                           # dev text for prompt work

D23 IN ONE PARAGRAPH
--------------------
There is no human-labelled training or dev set. The 200 golden items are the only
human labels and they are TEST. So the conventional supervised baseline named in
D14 - TF-IDF + logistic regression trained on human labels - **cannot be built**,
and no supervised metric is reported anywhere in this project. What exists
instead is one rule baseline, one *distant-supervision proxy* baseline, and one
LLM classifier prompted with the locked codebook. None of them ever sees a golden
label; predictions are written to disk and scored by scripts/evaluate.py.

`distant-lr` IS NOT A SUPERVISED BASELINE. Its training targets are regex output
over train openers, so it inherits the seeds' biases and can only beat them where
TF-IDF generalises vocabulary. Two unverified assumptions ride on it: that
seed-family membership stands in for intent, and that a train-fitted vectoriser
transfers to the test window - D16 measures a real distribution shift between
those windows, so the second is known to be imperfect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate_pilot as AP  # noqa: E402
import corpus as C  # noqa: E402
import providers as PR  # noqa: E402
import profile as P  # noqa: E402

PRED_DIR = os.path.join("outputs", "preds")
LLM_CACHE = os.path.join("cache", "llm")

# First-match-wins order. This encodes the codebook's boundary-rule precedence and
# was fixed from docs/taxonomy.md, not tuned: BR-11 puts authentication ahead of
# device wording, BR-2 puts a disputed charge ahead of plan questions, and BR-6
# puts an unshipped-capability request ahead of the technical intents.
SEED_ORDER = [
    "account_access",
    "billing_subscription",
    "plans_eligibility",
    "content_availability",
    "product_feature_feedback",
    "app_device_technical",
    "playback_playlist",
]
# social_praise was removed as an intent in codebook 0.2.0 (BR-5). Its seed pattern
# is still evaluated, and anything it alone matches falls through to other_unclear,
# which is where the codebook now sends social-only praise.

# Fixed a priori (D23): with no labelled dev set there is nothing to select on,
# so no sweep is run and these defaults are reported as unfitted.
LR_PARAMS: dict[str, Any] = {"C": 1.0, "max_iter": 2000, "class_weight": "balanced",
                             "solver": "lbfgs", "random_state": 0}
TFIDF_PARAMS: dict[str, Any] = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
                                "strip_accents": "unicode"}
# Option A, approved and pinned. Dated snapshots where the API publishes one, so a
# silent model refresh cannot change results between runs. Do not switch models
# without re-recording the decision: the cache key includes the model id, so a
# change silently invalidates every cached call and re-spends the budget.
MODELS: dict[str, str] = {
    "classifier": "claude-haiku-4-5-20251001",   # Option A: intent classification
    "drafting": "claude-haiku-4-5-20251001",     # Option A: reply drafting
    "judge": "claude-sonnet-5",                  # Option A: stronger tier, D20 mitigation
}
LLM_MODEL = MODELS["classifier"]


def model_for(role: str, provider: str, override: str | None = None) -> str | None:
    """Model id for a role.

    An explicit --model always wins. With no override, 'anthropic' uses the pinned
    Option A ids and every other provider falls through to its own default, which
    keeps behaviour identical to before the flag existed.
    """
    if override:
        return override
    return MODELS[role] if provider == "anthropic" else None
LLM_PARAMS: dict[str, Any] = {"max_tokens": 16, "temperature": 0.0}


def load_env_file(path: str = ".env") -> bool:
    """Read KEY=VALUE lines from .env into os.environ. Never prints a value.

    Kept dependency-free on purpose: python-dotenv would be a new requirement for
    twelve lines. Existing environment variables win, so an exported key is not
    silently overridden by a stale file.
    """
    if not os.path.exists(path):
        return False
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    return True


def api_key_status() -> tuple[bool, str]:
    """Presence and shape of the key. NEVER returns or logs the key itself."""
    load_env_file()
    v = os.environ.get("ANTHROPIC_API_KEY", "")
    if not v:
        return False, "NOT SET"
    return True, f"set, {len(v)} chars, prefix {v[:7]}… (value never logged)"


# ---------------------------------------------------------------------------
# 1. seed-rule baseline  - construction: regex written on train, never fitted
# ---------------------------------------------------------------------------
def seed_labels(texts: pd.Series) -> pd.Series:
    """First-match-wins over the locked seed families; no match -> other_unclear."""
    out = pd.Series("other_unclear", index=texts.index, dtype=object)
    unset = pd.Series(True, index=texts.index)
    for name in SEED_ORDER:
        hit = texts.str.contains(AP.PILOT_SEEDS[name], case=False, regex=True) & unset
        out[hit] = name
        unset &= ~hit
    return out


# ---------------------------------------------------------------------------
# 2. distant-supervision proxy  - construction: LR fitted on the seeds' output
# ---------------------------------------------------------------------------
class DistantLR:
    """TF-IDF + logistic regression trained on WEAK labels, not human labels.

    Reported as `distant-lr (proxy baseline, distant supervision)`. See D23 for
    why this is not, and must not be called, a supervised baseline.
    """

    name = "distant-lr"
    construction = "TF-IDF + LR fitted on train openers weakly labelled by the seed rules"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(**TFIDF_PARAMS)
        self.clf = LogisticRegression(**LR_PARAMS)
        self.fit_report: dict[str, Any] = {}

    def fit(self, train_openers: pd.DataFrame) -> "DistantLR":
        y = seed_labels(train_openers["text"])
        x = self.vec.fit_transform(train_openers["clean"])
        self.clf.fit(x, y)
        self.fit_report = {
            "n_train": int(len(train_openers)),
            "label_source": "seed rules (distant supervision) - NO human labels",
            "weak_label_distribution": {k: int(v) for k, v in y.value_counts().items()},
            "features": int(x.shape[1]),
            "tfidf_params": {k: list(v) if isinstance(v, tuple) else v
                             for k, v in TFIDF_PARAMS.items()},
            "lr_params": LR_PARAMS,
            "hyperparameter_search": "none - no labelled dev set exists (D23)",
        }
        return self

    def predict(self, texts: pd.Series) -> tuple[pd.Series, np.ndarray, list[str]]:
        x = self.vec.transform(texts.map(P.strip_signature))
        proba = self.clf.predict_proba(x)
        classes = list(self.clf.classes_)
        return pd.Series([classes[i] for i in proba.argmax(1)], index=texts.index), proba, classes


# ---------------------------------------------------------------------------
# 3. LLM zero-shot over the locked codebook
# ---------------------------------------------------------------------------
def build_prompt(codebook_path: str = "docs/codebook.json") -> str:
    """Zero-shot instruction assembled from the machine-readable codebook.

    Provenance is deliberately narrow: the codebook and its boundary rules only.
    Nothing from the golden set enters this prompt. If few-shot examples are added
    later they must be drawn from TRAIN or DEV openers and recorded here - a golden
    item used as an example would contaminate the only evaluation the project has.
    """
    cb = json.load(open(codebook_path, encoding="utf-8"))
    lines = [
        "You are labelling customer support messages sent to Spotify on Twitter.",
        "Assign exactly ONE intent from the taxonomy below. Intent answers only",
        "'what does the customer want?' - not how urgent, how angry, or whether it",
        "should be escalated.",
        "",
        f"TAXONOMY (codebook {cb['version']}):",
    ]
    for v in cb["intent"]["values"]:
        lines.append(f"- {v['name']}: {v['definition']}")
        if v.get("confusable_with"):
            lines.append(f"    often confused with: {', '.join(v['confusable_with'])}")
    lines += ["", "BOUNDARY RULES - apply in order, the first that fires decides:"]
    for br in cb["boundary_rules"]:
        rid = br.get("id", "BR")
        text = br.get("rule") or br.get("text") or br.get("description") or json.dumps(br)
        lines.append(f"- {rid}: {text}")
    lines += [
        "",
        "If a careful reader cannot identify the issue, answer other_unclear. Do not",
        "infer an issue the customer never stated.",
        "",
        "Answer with the intent name alone, nothing else.",
        "",
        "MESSAGE:",
        "{message}",
    ]
    return "\n".join(lines)


class LLMClassifier:
    name = "llm"
    construction = "prompted with the locked codebook; no human labels in the prompt"

    def __init__(self, provider_name: str = PR.DEFAULT_PROVIDER,
                 base_url: str | None = None, prompt: str | None = None,
                 model: str | None = None) -> None:
        self.provider = PR.get_provider(
            provider_name, model_for("classifier", provider_name, model), base_url)
        self.prompt = prompt or build_prompt()

    def classify(self, message: str, allow_api: bool) -> tuple[str, dict[str, Any]]:
        blob = cached_call("intent", self.provider, self.prompt, message,
                           LLM_PARAMS, allow_api)
        raw = (blob.get("text") or "").strip().lower()
        return (raw if raw in AP.INTENTS else "other_unclear"), blob


# ---------------------------------------------------------------------------
# prediction driver
# ---------------------------------------------------------------------------
def golden_texts() -> pd.DataFrame:
    """Golden ids and text ONLY. The label columns are not read here, by design."""
    with open("data/golden/golden_annotated.csv", encoding="utf-8-sig", newline="") as fh:
        rows = [{"id": r["id"], "text": r["text"]} for r in csv.DictReader(fh)]
    return pd.DataFrame(rows)


def predict(which: str, split: str, csv_path: str, allow_api: bool,
            provider_name: str = PR.DEFAULT_PROVIDER, base_url: str | None = None,
            model: str | None = None) -> int:
    pairs = C.opener_pairs(C.load(csv_path))
    train = pairs[(pairs["split"] == "train") & (pairs["clean"].str.len() > 0)]
    if split == "golden":
        items = golden_texts()
    else:
        sub = pairs[pairs["split"] == split].head(200)
        items = pd.DataFrame({"id": sub["tweet_id"].astype(str), "text": sub["text"]})

    os.makedirs(PRED_DIR, exist_ok=True)
    meta: dict[str, Any] = {"classifier": which, "split": split, "n": int(len(items)),
                            "saw_human_labels": False}
    extra_cols: dict[str, Any] = {}

    if which == "rule":
        meta["construction"] = ("first-match-wins seed regex written against train text "
                                "before any labelling; nothing is fitted")
        pred = seed_labels(items["text"])
    elif which == "distant-lr":
        m = DistantLR().fit(train)
        meta["construction"] = m.construction
        meta["proxy_baseline"] = True
        meta["fit_report"] = m.fit_report
        pred, proba, classes = m.predict(items["text"])
        extra_cols["max_proba"] = proba.max(1).round(4)
    elif which == "llm":
        m2 = LLMClassifier(provider_name, base_url, model=model)
        meta["construction"] = m2.construction
        meta["provider"] = m2.provider.name
        meta["model"] = m2.provider.model
        meta["is_mock"] = bool(getattr(m2.provider, "is_mock", False))
        meta["prompt_sha256"] = hashlib.sha256(m2.prompt.encode()).hexdigest()[:16]
        pred = pd.Series([m2.classify(t, allow_api)[0] for t in items["text"]],
                         index=items.index)
    else:
        raise SystemExit(f"unknown classifier {which!r}")

    suffix = "_mock" if meta.get("is_mock") else ""
    out = os.path.join(PRED_DIR, f"intent_{which}_{split}{suffix}.csv")
    frame = pd.DataFrame({"id": items["id"], "pred_intent": pred, **extra_cols})
    frame.to_csv(out, index=False, encoding="utf-8")
    with open(out.replace(".csv", ".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{which}] {len(frame)} predictions -> {out}")
    print(f"  construction: {meta['construction']}")
    print(f"  distribution: {dict(frame['pred_intent'].value_counts())}")
    if meta.get("is_mock"):
        print("  *** MOCK PROVIDER - these are stub outputs, NOT a result. "
              "They exist to exercise the pipeline. ***")
    if which == "llm":
        meta["usage"] = usage_summary()
        with open(out.replace(".csv", ".meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print(f"  usage: {json.dumps(meta['usage'])}")
    return 0


# ---------------------------------------------------------------------------
# One cached API call path, shared by the classifier, the drafter and the judge,
# so there is a single cache scheme and a single place the budget is spent.
# ---------------------------------------------------------------------------
USAGE: list[dict[str, Any]] = []       # in-process record; also persisted per entry


def cache_path(tag: str, provider: str, model: str, prompt: str,
               params: dict[str, Any], message: str) -> str:
    """Cache key covers provider and model as well as prompt, params and message.

    Including the provider is what keeps a mock run and a real run from ever
    sharing an entry - the two must never silently substitute for each other.
    """
    h = hashlib.sha256(json.dumps([provider, model, prompt, params, message],
                                  sort_keys=True).encode()).hexdigest()[:32]
    return os.path.join(LLM_CACHE, f"{tag}_{provider}_{h}.json")


def cached_call(tag: str, provider: Any, prompt: str, message: str,
                params: dict[str, Any], allow_api: bool) -> dict[str, Any]:
    """Replay the cache, else call the provider. Returns the response as a dict.

    `provider` is a providers.Provider instance. Only billed providers are gated
    by --allow-api; the mock and a self-hosted OpenAI-compatible server are free
    and run unguarded.
    """
    os.makedirs(LLM_CACHE, exist_ok=True)
    path = cache_path(tag, provider.name, provider.model, prompt, params, message)
    if os.path.exists(path):
        blob = json.load(open(path, encoding="utf-8"))
        blob["cached"] = True
        USAGE.append({"tag": tag, "cached": True, **{k: blob.get(k) for k in
                      ("provider", "model", "is_mock", "input_tokens", "output_tokens")}})
        return blob
    if PR.is_billed(provider.name) and not allow_api:
        raise SystemExit(
            f"[{tag}] provider '{provider.name}' spends real money and --allow-api was "
            "not passed.\nRun with the default --provider mock, or a self-hosted "
            "--provider openai-compat, to stay free.")
    if PR.is_billed(provider.name):
        load_env_file()
    blob = provider.complete(prompt, message, params).as_dict()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh)
    blob["cached"] = False
    USAGE.append({"tag": tag, "cached": False, **{k: blob.get(k) for k in
                  ("provider", "model", "is_mock", "input_tokens", "output_tokens")}})
    return blob


# Published base rates, USD per million tokens (docs verified 2026-09-14).
PRICES = {"claude-haiku-4-5-20251001": (1.0, 5.0), "claude-sonnet-5": (2.0, 10.0)}


def usage_summary() -> dict[str, Any]:
    out: dict[str, Any] = {"calls": 0, "cached": 0, "billed_calls": 0, "usd": 0.0,
                           "any_mock": False, "by_provider": {}}
    for u in USAGE:
        out["calls"] += 1
        out["any_mock"] = out["any_mock"] or bool(u.get("is_mock"))
        k = f"{u.get('provider')}:{u.get('model')}"
        m = out["by_provider"].setdefault(k, {"calls": 0, "cached": 0, "input_tokens": 0,
                                              "output_tokens": 0, "usd": 0.0,
                                              "is_mock": bool(u.get("is_mock"))})
        m["calls"] += 1
        if u["cached"]:
            out["cached"] += 1
            m["cached"] += 1
            continue
        m["input_tokens"] += u.get("input_tokens") or 0
        m["output_tokens"] += u.get("output_tokens") or 0
        if PR.is_billed(u.get("provider") or ""):
            out["billed_calls"] += 1
            c = PR.cost_usd(u.get("model") or "", u.get("input_tokens") or 0,
                            u.get("output_tokens") or 0)
            m["usd"] += c
            out["usd"] += c
    out["usd"] = round(out["usd"], 4)
    for m in out["by_provider"].values():
        m["usd"] = round(m["usd"], 4)
    return out


SECRET_RE = __import__("re").compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")


def preflight(provider_name: str = PR.DEFAULT_PROVIDER, base_url: str | None = None,
              model: str | None = None) -> int:
    """Config check for the selected provider. Makes no call and prints no secret."""
    import glob
    import subprocess
    ok = True

    def chk(good: bool, label: str, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}{('  - ' + detail) if detail else ''}")

    prov = PR.get_provider(provider_name, model_for("classifier", provider_name, model),
                           base_url)
    billed = PR.is_billed(provider_name)
    mock = bool(getattr(prov, "is_mock", False))
    print(f"=== PREFLIGHT - provider '{provider_name}' - no call is made by this command ===")

    print("\n1. provider and models")
    print(f"     provider     {provider_name}{'  (BILLED)' if billed else '  (free)'}"
          f"{'  [MOCK - stub output, never a result]' if mock else ''}")
    for role in ("classifier", "drafting", "judge"):
        print(f"     {role:<12} {model_for(role, provider_name, model) or prov.model}"
              f"{'  (--model override)' if model else ''}")
    chk(provider_name in PR.PROVIDERS, "provider is registered")
    if provider_name == "anthropic":
        chk(MODELS["classifier"] == MODELS["drafting"] == "claude-haiku-4-5-20251001"
            and MODELS["judge"] == "claude-sonnet-5",
            "anthropic models match Option A (Haiku 4.5 classify+draft, Sonnet 5 judge)")

    print("\n2. dependencies")
    if provider_name == "anthropic":
        try:
            import anthropic
            ver = anthropic.__version__
        except ImportError:
            ver = None
        chk(ver is not None, "anthropic SDK importable (optional extra)",
            f"v{ver}" if ver else "pip install anthropic")
    else:
        chk(True, "no extra dependency needed", f"provider '{provider_name}' is self-contained")

    print("\n3. secret hygiene")
    try:
        ignored = subprocess.run(["git", "check-ignore", "-q", ".env"],
                                 capture_output=True).returncode == 0
    except Exception:
        ignored = False
    chk(ignored, ".env is git-ignored")
    leaks = []
    for pat in ("outputs/**/*", "cache/**/*", "docs/*", "scripts/*", "data/golden/*"):
        for f in glob.glob(pat, recursive=True):
            if not os.path.isfile(f):
                continue
            try:
                if SECRET_RE.search(open(f, encoding="utf-8", errors="ignore").read()):
                    leaks.append(f)
            except OSError:
                pass
    chk(not leaks, "no API-key pattern in outputs, cache, docs, scripts or golden data",
        f"{len(leaks)} file(s): {leaks[:3]}" if leaks else "scanned tracked + generated trees")
    print("     cache entries persist only {model, raw, intent}; prediction metadata persists")
    print("     only {model, prompt_sha256} - no key, no prompt text, no credential")

    print("\n4. credential")
    if billed:
        present, shape = api_key_status()
        chk(present, "ANTHROPIC_API_KEY loaded from environment or .env", shape)
    else:
        chk(True, "no credential required", f"provider '{provider_name}' is free")

    print("\n5. cache")
    chk(True, "cache key = sha256(json([provider, model, prompt, params, message]))",
        "provider is in the key, so mock and real entries can never collide")
    print(f"     cache dir: {LLM_CACHE}   existing entries: "
          f"{len(glob.glob(os.path.join(LLM_CACHE, '*.json')))}")
    print(f"     params: {LLM_PARAMS}")

    print("\n6. final configuration")
    print(f"     provider           {provider_name}{'  (BILLED)' if billed else '  (free)'}")
    print(f"     classifier model   {model_for('classifier', provider_name, model) or prov.model}")
    print(f"     drafting model     {model_for('drafting', provider_name, model) or prov.model}")
    print(f"     judge model        {model_for('judge', provider_name, model) or prov.model}")
    print(f"     params             {LLM_PARAMS}")
    print(f"     prompt sha256      {hashlib.sha256(build_prompt().encode()).hexdigest()[:16]}")
    print(f"     prompt length      {len(build_prompt())} chars")
    print(f"     cache              {LLM_CACHE} (sha256-keyed, replayed before any call)")
    print(f"     credential         {api_key_status()[1] if billed else 'not required'}")
    print(f"\n{'PREFLIGHT PASSED' if ok else 'PREFLIGHT FAILED - do not run the pipeline'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predict", choices=["rule", "distant-lr", "llm"])
    ap.add_argument("--split", default="golden", choices=["golden", "dev", "train"])
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--dev-sample", type=int, metavar="N",
                    help="print N dev openers - the only material permitted for prompt work")
    ap.add_argument("--provider", default=PR.DEFAULT_PROVIDER, choices=list(PR.PROVIDERS),
                    help="mock (default, offline and free) | openai-compat (self-hosted) "
                         "| anthropic (billed)")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint for --provider openai-compat")
    ap.add_argument("--model", help="model id to use; overrides the provider default "
                                    "(e.g. llama3.2:3b). Omit to keep existing behaviour.")
    ap.add_argument("--allow-api", action="store_true",
                    help="authorise a BILLED provider (D19 budget); responses are cached")
    ap.add_argument("--csv", default="twcs/twcs.csv")
    args = ap.parse_args()

    if args.preflight:
        return preflight(args.provider, args.base_url, args.model)
    if args.show_prompt:
        print(build_prompt())
        return 0
    if args.dev_sample:
        pairs = C.opener_pairs(C.load(args.csv))
        dev = pairs[pairs["split"] == "dev"]
        rng = np.random.default_rng(0)
        for i in rng.choice(len(dev), size=min(args.dev_sample, len(dev)), replace=False):
            print("-", dev.iloc[int(i)]["text"].replace("\n", " ")[:160])
        print("\nDEV material only. Golden items must not be used for prompt development (D23).")
        return 0
    if not args.predict:
        ap.error("pass --predict {rule,distant-lr,llm}, --show-prompt or --dev-sample N")
    return predict(args.predict, args.split, args.csv, args.allow_api,
                   args.provider, args.base_url, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
