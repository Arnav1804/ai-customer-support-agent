"""Small OpenAI-compatible chat client with one retry and no SDK dependency."""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from common.env import load_dotenv


class LLMError(RuntimeError):
    """A recoverable API, configuration, or response error."""


class ChatClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass
class OpenAICompatibleClient:
    api_key: str
    model: str
    base_url: str
    timeout_seconds: int = 45

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleClient":
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set. Add it to .env or your environment.")
        return cls(
            api_key=api_key,
            model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    parsed = json.load(response)
                content = parsed["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise KeyError("empty chat-completion content")
                return content.strip()
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
                logging.warning("LLM request failed (attempt %d of 2): %s", attempt + 1, exc)
                if attempt == 0:
                    time.sleep(0.5)
        raise LLMError(f"LLM request failed after one retry: {last_error}")


@dataclass
class UnavailableClient:
    """Lets batch mode record skipped rows when API setup is unavailable."""
    reason: str

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMError(self.reason)
