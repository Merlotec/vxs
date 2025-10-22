from __future__ import annotations
import os
import re
import requests
from typing import Any, Dict, Optional


class LLMBackend:
    def generate(self, *, system: str, user: str, model: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
        raise NotImplementedError


def _extract_code_block(text: str) -> str:
    """Extract first Python code block if present; else return full text."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


class LLVPNAnthropicClient(LLMBackend):
    """
    Thin wrapper around an LLVPN proxy for Anthropic's Messages API.

    Expects an env var for the API key (default: LLVPN_API_KEY) and posts to the given URL.
    Body uses the standard Anthropic messages schema.
    """

    def __init__(self, *, url: str, api_key_env: str = "LLVPN_API_KEY", anthropic_version: str = "2023-06-01") -> None:
        self.url = url
        self.api_key_env = api_key_env
        self.anthropic_version = anthropic_version

    def generate(self, *, system: str, user: str, model: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env var {self.api_key_env}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Some Anthropic implementations require this header; proxy may forward/ignore.
            "anthropic-version": self.anthropic_version,
        }

        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                    ],
                }
            ],
        }

        resp = requests.post(self.url, json=body, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        # Anthropic messages response typically: { content: [{ type: 'text', text: '...' }], ... }
        try:
            content = data["content"][0]["text"]
        except Exception:
            # Best-effort fallback across proxy variations
            content = data.get("completion") or data.get("output") or str(data)
        return _extract_code_block(content)


class LLVPNOpenAIClient(LLMBackend):
    """
    Thin wrapper around an LLVPN proxy for OpenAI's Chat Completions API.

    Expects an env var for the API key (default: LLVPN_API_KEY) and posts to the given URL.
    Body uses the standard OpenAI chat.completions schema.
    """

    def __init__(self, *, url: str, api_key_env: str = "LLVPN_API_KEY") -> None:
        self.url = url
        self.api_key_env = api_key_env

    def generate(self, *, system: str, user: str, model: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env var {self.api_key_env}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        resp = requests.post(self.url, json=body, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        # OpenAI chat completions: choices[0].message.content
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = data.get("content") or data.get("completion") or str(data)
        return _extract_code_block(content)
