from __future__ import annotations

import json
import os
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml


def _gemini_http_timeout_seconds() -> tuple[float, float]:
    """
    (connect, read) timeouts for Gemini generateContent.
    Override with GEMINI_HTTP_TIMEOUT_READ (seconds), default 600 read / 30 connect.
    """
    read_s = 600.0
    raw = (os.environ.get("GEMINI_HTTP_TIMEOUT_READ") or "").strip()
    if raw:
        try:
            read_s = max(60.0, float(raw))
        except ValueError:
            pass
    return (30.0, read_s)


def _gemini_json_looks_truncated(text: str) -> bool:
    """
    Heuristic: model hit max tokens mid-string, producing invalid JSON.
    """
    t = (text or "").strip()
    if not t:
        return False
    t = re.sub(r"```json|```", "", t, flags=re.IGNORECASE).strip()
    if not t or t[0] not in "[{":
        return False
    try:
        json.loads(t)
        return False
    except Exception:
        return True


@dataclass
class ModelEndpoint:
    provider: str
    model: str
    api_key: str = ""
    base_url: str | None = None
    model_cfg: dict[str, Any] = field(default_factory=dict)
    provider_cfg: dict[str, Any] = field(default_factory=dict)


class ModelRouter:
    """
    Routes logical model IDs to concrete provider endpoints.

    Supports:
    - anthropic
    - openai / openai_compatible (Chat Completions API)
    """

    def __init__(self, models: dict[str, ModelEndpoint], defaults: dict[str, Any] | None = None):
        self.models = models
        self.defaults = defaults or {}
        self._anthropic_clients: dict[tuple[str, str | None], Any] = {}

    @classmethod
    def from_config_file(cls, path: str | None) -> "ModelRouter":
        if not path:
            return cls(models={})
        p = Path(path)
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid model config (expected object): {path}")

        defaults_raw = cfg.get("defaults") or {}
        providers_raw = cfg.get("providers") or {}
        models_raw = cfg.get("models") or {}
        judge_raw = cfg.get("judge") or {}
        if not isinstance(defaults_raw, dict) or not isinstance(providers_raw, dict) or not isinstance(models_raw, dict):
            raise ValueError("Model config must contain 'defaults', 'providers', and 'models' maps.")

        # Merge agent models + judge models so both are resolvable
        all_models_raw = dict(models_raw)
        if isinstance(judge_raw, dict):
            for model_id, model_cfg in judge_raw.items():
                if isinstance(model_cfg, dict) and model_id not in all_models_raw:
                    all_models_raw[model_id] = model_cfg

        models: dict[str, ModelEndpoint] = {}
        for model_id, model_cfg in all_models_raw.items():
            if not isinstance(model_cfg, dict):
                continue
            provider_name = str(model_cfg.get("provider") or "").strip()
            if not provider_name:
                raise ValueError(f"Model '{model_id}' is missing provider.")
            provider_defaults = providers_raw.get(provider_name, {})
            if not isinstance(provider_defaults, dict):
                provider_defaults = {}

            model_name = str(model_cfg.get("model") or "").strip()
            if not model_name:
                raise ValueError(f"Model '{model_id}' is missing model name.")

            models[str(model_id)] = ModelEndpoint(
                provider=provider_name.strip().lower(),
                model=model_name,
                model_cfg=dict(model_cfg),
                provider_cfg=dict(provider_defaults),
            )
        return cls(models=models, defaults=defaults_raw)

    def has_model(self, model_id: str) -> bool:
        return model_id in self.models

    def resolve_endpoint(self, model_id: str) -> ModelEndpoint:
        if model_id not in self.models:
            raise KeyError(
                f"Model ID '{model_id}' not found in model config. "
                "Use --model-config with matching models.<id> entries."
            )
        endpoint = self.models[model_id]
        return ModelEndpoint(
            provider=endpoint.provider,
            model=endpoint.model,
            api_key=_resolve_api_key(endpoint.model_cfg, endpoint.provider_cfg),
            base_url=_resolve_base_url(endpoint.model_cfg, endpoint.provider_cfg),
            model_cfg=endpoint.model_cfg,
            provider_cfg=endpoint.provider_cfg,
        )

    def generate(
        self,
        model_id: str,
        system: str,
        user: str,
        max_tokens: int,
        cache_system_prompt: bool = False,
    ) -> str:
        endpoint = self.models[model_id]
        resolved_endpoint = self.resolve_endpoint(model_id)
        provider = endpoint.provider
        if provider == "anthropic":
            return self._generate_anthropic(
                resolved_endpoint,
                system,
                user,
                max_tokens,
                cache_system_prompt=cache_system_prompt,
            )
        # Google: use native Gemini API with safetySettings support
        if provider == "google":
            return self._generate_gemini_native(resolved_endpoint, system, user, max_tokens)
        # Many providers expose OpenAI-compatible chat endpoints.
        if provider in {
            "openai",
            "openai_compatible",
            "groq",
            "mistral",
            "deepseek",
            "together",
        }:
            return self._generate_openai_compatible(resolved_endpoint, system, user, max_tokens)
        raise ValueError(f"Unsupported provider '{provider}' for model '{model_id}'.")

    def _generate_anthropic(
        self,
        endpoint: ModelEndpoint,
        system: str,
        user: str,
        max_tokens: int,
        cache_system_prompt: bool = False,
    ) -> str:
        try:
            from anthropic import Anthropic
        except Exception:
            # Fallback: call Anthropic Messages HTTP API directly so the harness
            # can run even when the anthropic SDK is unavailable.
            return self._generate_anthropic_http(
                endpoint,
                system,
                user,
                max_tokens,
                cache_system_prompt=cache_system_prompt,
            )

        cache_key = (endpoint.api_key, endpoint.base_url)
        client = self._anthropic_clients.get(cache_key)
        if client is None:
            kwargs: dict[str, Any] = {"api_key": endpoint.api_key}
            if endpoint.base_url:
                kwargs["base_url"] = endpoint.base_url
            client = Anthropic(**kwargs)
            self._anthropic_clients[cache_key] = client

        system_payload: str | list[dict[str, Any]]
        if cache_system_prompt:
            system_payload = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_payload = system

        resp = client.messages.create(
            model=endpoint.model,
            max_tokens=max_tokens,
            system=system_payload,
            messages=[{"role": "user", "content": user}],
        )
        return str(resp.content[0].text or "")

    def _generate_anthropic_http(
        self,
        endpoint: ModelEndpoint,
        system: str,
        user: str,
        max_tokens: int,
        cache_system_prompt: bool = False,
    ) -> str:
        base = (endpoint.base_url or "https://api.anthropic.com").rstrip("/")
        url = f"{base}/v1/messages"
        headers = {
            "x-api-key": endpoint.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if cache_system_prompt:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
            system_payload: str | list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_payload = system
        payload = {
            "model": endpoint.model,
            "max_tokens": max_tokens,
            "system": system_payload,
            "messages": [{"role": "user", "content": user}],
        }
        resp = _post_with_retries(url=url, headers=headers, payload=payload, provider_label="Anthropic HTTP")
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Anthropic HTTP request failed ({resp.status_code}): {resp.text[:500]}"
            )
        data = resp.json()
        content = data.get("content") or []
        if not isinstance(content, list):
            return ""
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        return "\n".join(text_parts).strip()

    def _generate_gemini_native(
        self, endpoint: ModelEndpoint, system: str, user: str, max_tokens: int
    ) -> str:
        """
        Use Google's native Generative AI API (generateContent) which supports
        safetySettings. The OpenAI-compatible endpoint does NOT support safety
        settings, causing empty responses on code review tasks.
        """
        # Native endpoint: /v1beta/models/{model}:generateContent
        base = "https://generativelanguage.googleapis.com"
        url = f"{base}/v1beta/models/{endpoint.model}:generateContent?key={endpoint.api_key}"
        headers = {"Content-Type": "application/json"}
        safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        ]

      
        effective_tokens = max(int(max_tokens), 8192)
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": user}]},
            ],
            "systemInstruction": {"parts": [{"text": system}]},
            "safetySettings": safety_settings,
            "generationConfig": {
                "maxOutputTokens": effective_tokens,
                "temperature": 0,
            },
        }

        content, finish_reason = self._do_gemini_native_request(url, headers, payload)
        if not (content or "").strip():
            content, finish_reason = self._do_gemini_native_request(url, headers, payload)
        elif _gemini_json_looks_truncated(content):
            payload["generationConfig"]["maxOutputTokens"] = min(max(effective_tokens * 2, 16384), 65536)
            content, _finish2 = self._do_gemini_native_request(url, headers, payload)
        if not (content or "").strip():
            return "[]"
        return content

    def _do_gemini_native_request(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> tuple[str, str | None]:
        """Execute a single request to Gemini's native generateContent API."""
      
        gemini_timeout = _gemini_http_timeout_seconds()
        resp = _post_with_retries(
            url=url,
            headers=headers,
            payload=payload,
            provider_label="Gemini native",
            timeout=gemini_timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gemini native request failed ({resp.status_code}): {resp.text[:500]}"
            )
        data = resp.json()

        if data.get("promptFeedback", {}).get("blockReason"):
            return "", None

        candidates = data.get("candidates") or []
        if not candidates:
            return "", None
        c0 = candidates[0]
        finish_reason = c0.get("finishReason")
        content = c0.get("content") or {}
        parts = content.get("parts") or []
        text_parts = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "\n".join(text_parts).strip(), finish_reason if isinstance(finish_reason, str) else None

    def _generate_openai_compatible(
        self, endpoint: ModelEndpoint, system: str, user: str, max_tokens: int
    ) -> str:
        base = (endpoint.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json",
        }
        token_key = "max_completion_tokens" if endpoint.provider == "openai" else "max_tokens"
        payload = {
            "model": endpoint.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            token_key: max_tokens,
            "temperature": 0,
        }
        return self._do_openai_request(url, headers, payload, endpoint.provider) or ""

    def _do_openai_request(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], provider: str
    ) -> str:
        resp = _post_with_retries(
            url=url, headers=headers, payload=payload, provider_label="OpenAI-compatible"
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OpenAI-compatible request failed ({resp.status_code}): {resp.text[:500]}"
            )
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    t = block.get("text")
                    if isinstance(t, str):
                        text_parts.append(t)
            return "\n".join(text_parts).strip()
        return str(content or "")


def _resolve_api_key(model_cfg: dict[str, Any], provider_cfg: dict[str, Any]) -> str:
    if isinstance(model_cfg.get("api_key"), str) and model_cfg.get("api_key").strip():
        return model_cfg["api_key"].strip()
    if isinstance(provider_cfg.get("api_key"), str) and provider_cfg.get("api_key").strip():
        return provider_cfg["api_key"].strip()

    model_env = model_cfg.get("api_key_env")
    provider_env = provider_cfg.get("api_key_env")
    env_name = model_env if isinstance(model_env, str) and model_env.strip() else provider_env
    if isinstance(env_name, str) and env_name.strip():
        env_name = env_name.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", env_name):
            return env_name

        value = (os.environ.get(env_name) or "").strip()
        if value:
            return value
        raise RuntimeError(
            f"Missing API key env var: {env_name}. "
            "If you pasted a raw key, use 'api_key' (or keep it in .env and reference with api_key_env)."
        )

    raise RuntimeError("Missing API key configuration (api_key or api_key_env).")


def _resolve_base_url(model_cfg: dict[str, Any], provider_cfg: dict[str, Any]) -> str | None:
    model_base = model_cfg.get("base_url")
    if isinstance(model_base, str) and model_base.strip():
        return model_base.strip()
    provider_base = provider_cfg.get("base_url")
    if isinstance(provider_base, str) and provider_base.strip():
        return provider_base.strip()
    return None


def _post_with_retries(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    provider_label: str,
    timeout: float | tuple[float, float] = 120,
    max_attempts: int = 6,
) -> requests.Response:
    last_error: Exception | None = None
    last_response: requests.Response | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        except requests.RequestException as e:
            last_error = e
            if attempt == max_attempts:
                raise RuntimeError(
                    f"{provider_label} request failed after {max_attempts} attempts: {e}"
                ) from e
            time.sleep(_retry_sleep_seconds(None, attempt))
            continue

        if resp.status_code < 400:
            return resp

        last_response = resp
        if resp.status_code not in {408, 409, 425, 429, 500, 502, 503, 504}:
            return resp
        if attempt == max_attempts:
            return resp
        time.sleep(_retry_sleep_seconds(resp, attempt))

    if last_response is not None:
        return last_response
    raise RuntimeError(f"{provider_label} request failed with unknown error: {last_error}")


def _retry_sleep_seconds(resp: requests.Response | None, attempt: int) -> float:
    if resp is not None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                wait = float(retry_after.strip())
                return max(1.0, min(wait + 0.25, 60.0))
            except Exception:
                pass
    # bounded exponential backoff
    return min(2 ** min(attempt, 5), 20)

