#!/usr/bin/env python3
"""LLM provider abstraction. Default is a deterministic mock: no key, no network.

    from providers import get_provider
    p = get_provider("mock")                 # default; offline, free, deterministic
    p = get_provider("openai-compat", base_url="http://127.0.0.1:11434/v1")
    p = get_provider("anthropic", model="claude-haiku-4-5-20251001")

WHY A MOCK IS THE DEFAULT, AND WHAT IT IS NOT
---------------------------------------------
The pipeline must run end to end on a clean clone with no credits, so every LLM
stage needs *something* to call. The mock supplies that.

**The mock does not simulate a language model and its output is not a result.**
It is a deterministic stub whose only jobs are to exercise the plumbing - prompt
assembly, retrieval wiring, guardrails, routing, caching, artefact schemas - and
to keep those paths under test. Intent choices are a hash of the message, not a
judgement; judge scores are a function of surface features, not quality. Numbers
computed from mock output measure the harness, never the system.

Every mock response is stamped `is_mock=True`, every artefact it produces is
written with `is_mock: true` in its metadata and a `_mock` filename suffix, and
`scripts/evaluate.py` refuses to place mock-derived rows in a headline table.
Those three barriers exist so a mock run can never be mistaken for a real one.

LOCAL / FREE MODELS
-------------------
`openai-compat` targets any OpenAI-compatible server the user runs themselves -
Ollama, llama.cpp's `llama-server`, LM Studio, vLLM. No key, no per-token cost.
That is the supported free path to *real* model output.

It cannot be exercised inside this Cowork VM: 2 cores, 3.9 GB RAM, no GPU, and
`huggingface.co` and `ollama.com` are both blocked by the egress allowlist
(measured, not assumed - `pypi.org` answers 200, those two answer 000). So the
runtime can be installed here but no weights can be fetched. Run it on a machine
that can reach model weights and point `--base-url` at it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
from dataclasses import dataclass
from typing import Any

DEFAULT_PROVIDER = "mock"
PROVIDERS = ("mock", "openai-compat", "anthropic")

# Providers that spend real money. Anything here needs an explicit --allow-api.
BILLED = ("anthropic",)

# Published Anthropic base rates, USD per million tokens (docs checked 2026-09-14).
PRICES = {"claude-haiku-4-5-20251001": (1.0, 5.0), "claude-sonnet-5": (2.0, 10.0)}


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    provider: str
    model: str
    is_mock: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "provider": self.provider,
                "model": self.model, "is_mock": self.is_mock}


# ---------------------------------------------------------------------------
# mock
# ---------------------------------------------------------------------------
INTENTS = ["account_access", "billing_subscription", "plans_eligibility",
           "content_availability", "playback_playlist", "app_device_technical",
           "product_feature_feedback", "other_unclear"]

MOCK_DRAFT = ("Sorry for the trouble here. Try signing out and back in, and make sure the "
              "app is on the latest version. If it's still happening, send us a DM and "
              "we'll take a closer look.")


class MockProvider:
    """Offline stub. Deterministic in the message, and honest about being noise.

    No retry logic here on purpose: there is no network, no server and no failure
    mode to recover from. Retries belong to real providers only.
    """

    name = "mock"
    is_mock = True

    def __init__(self, model: str = "deterministic-stub-v1", **_: Any) -> None:
        self.model = model

    @staticmethod
    def _digest(message: str) -> int:
        return int(hashlib.sha256(message.encode()).hexdigest()[:8], 16)

    def complete(self, prompt: str, message: str, params: dict[str, Any]) -> Response:
        task = self._task_of(prompt)
        if task == "intent":
            # a hash, not a classification - see the module docstring
            text = INTENTS[self._digest(message) % len(INTENTS)]
        elif task == "judge":
            reply = message.split("PROPOSED REPLY:", 1)[-1]
            routes = bool(re.search(r"\bdms?\b|direct message", reply, re.I))
            brief = len(reply) <= 280
            score = 3 + int(routes) + int(brief)          # surface features, not quality
            text = json.dumps({"addresses_need": min(5, score), "grounded": min(5, score),
                               "routing": 5 if routes else 2, "tone": 4,
                               "acceptable": bool(routes and brief),
                               "why": "deterministic stub: surface features only"})
        else:
            text = MOCK_DRAFT
        return Response(text=text, input_tokens=len(prompt + message) // 4,
                        output_tokens=len(text) // 4, provider=self.name,
                        model=self.model, is_mock=True)

    @staticmethod
    def _task_of(prompt: str) -> str:
        if "Answer with JSON only" in prompt:
            return "judge"
        if "Assign exactly ONE intent" in prompt:
            return "intent"
        return "draft"


# ---------------------------------------------------------------------------
# local / self-hosted, OpenAI-compatible
# ---------------------------------------------------------------------------
class OpenAICompatProvider:
    """Any OpenAI-compatible endpoint: Ollama, llama-server, LM Studio, vLLM.

    No API key is required by default; a server that wants one reads it from
    OPENAI_API_KEY. Nothing here is billed by us.
    """

    name = "openai-compat"
    is_mock = False

    def __init__(self, model: str = "llama3.2:3b", base_url: str | None = None,
                 **_: Any) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or "http://127.0.0.1:11434/v1").rstrip("/")

    # Transient failures worth retrying: the server is briefly busy, loading a model
    # into RAM, or the socket timed out mid-generation on a slow CPU. Anything else -
    # a bad request, an unknown model, an auth refusal - is permanent and is raised
    # immediately rather than retried three times behind the user's back.
    ATTEMPTS = 3
    BACKOFF_SECONDS = (2.0, 5.0)          # waited before attempts 2 and 3
    RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    def complete(self, prompt: str, message: str, params: dict[str, Any]) -> Response:
        last: Exception | None = None
        for attempt in range(1, self.ATTEMPTS + 1):
            try:
                return self._once(prompt, message, params)
            except urllib.error.HTTPError as e:
                if e.code not in self.RETRY_STATUS:
                    raise SystemExit(
                        f"[openai-compat] HTTP {e.code} from {self.base_url} for model "
                        f"'{self.model}'. This is a permanent error and was not retried. "
                        f"Check the model name and that the server is serving it.") from e
                last = e
            except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
                last = e
            if attempt < self.ATTEMPTS:
                wait = self.BACKOFF_SECONDS[attempt - 1]
                print(f"  [openai-compat] attempt {attempt}/{self.ATTEMPTS} failed "
                      f"({type(last).__name__}: {last}); retrying in {wait:.0f}s",
                      file=__import__("sys").stderr)
                time.sleep(wait)
        raise SystemExit(
            f"[openai-compat] all {self.ATTEMPTS} attempts failed against {self.base_url} "
            f"(model '{self.model}'). Last error: {type(last).__name__}: {last}\n"
            f"Nothing was cached for this call. Is the server running and the model pulled?")

    def _once(self, prompt: str, message: str, params: dict[str, Any]) -> Response:
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt.replace("{message}", message)}],
            "temperature": params.get("temperature", 0.0),
            "max_tokens": params.get("max_tokens", 256),
        }).encode()
        headers = {"Content-Type": "application/json"}
        if os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", body, headers)
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.load(r)
        usage = data.get("usage") or {}
        return Response(text=data["choices"][0]["message"]["content"].strip(),
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        provider=self.name, model=self.model)


# ---------------------------------------------------------------------------
# anthropic (optional, billed)
# ---------------------------------------------------------------------------
class AnthropicProvider:
    name = "anthropic"
    is_mock = False

    def __init__(self, model: str = "claude-haiku-4-5-20251001", **_: Any) -> None:
        self.model = model
        self._client = None

    def complete(self, prompt: str, message: str, params: dict[str, Any]) -> Response:
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise SystemExit("provider 'anthropic' needs ANTHROPIC_API_KEY "
                                 "(put it in .env, which is git-ignored).")
            import anthropic
            self._client = anthropic.Anthropic()
        resp = self._client.messages.create(
            model=self.model, **params,
            messages=[{"role": "user", "content": prompt.replace("{message}", message)}])
        return Response(text=resp.content[0].text.strip(),
                        input_tokens=resp.usage.input_tokens,
                        output_tokens=resp.usage.output_tokens,
                        provider=self.name, model=self.model)


_REGISTRY = {"mock": MockProvider, "openai-compat": OpenAICompatProvider,
             "anthropic": AnthropicProvider}


def get_provider(name: str = DEFAULT_PROVIDER, model: str | None = None,
                 base_url: str | None = None):
    if name not in _REGISTRY:
        raise SystemExit(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")
    kwargs: dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    if base_url:
        kwargs["base_url"] = base_url
    return _REGISTRY[name](**kwargs)


def is_billed(name: str) -> bool:
    return name in BILLED


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    pi, po = PRICES.get(model, (0.0, 0.0))
    return in_tok / 1e6 * pi + out_tok / 1e6 * po
