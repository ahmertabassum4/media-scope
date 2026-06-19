from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI


class OpenRouterVisionClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        retries: int,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def classify(self, model: str, prompt: str, data_url: str) -> tuple[str, dict[str, Any]]:
        return self.complete_vision(model=model, prompt=prompt, data_urls=[data_url])

    def complete_vision(
        self,
        model: str,
        prompt: str,
        data_urls: list[str],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for data_url in data_urls:
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                params: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "timeout": self.timeout_seconds,
                }
                if temperature is not None:
                    params["temperature"] = temperature
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                if response_format is not None:
                    params["response_format"] = response_format
                response = self.client.chat.completions.create(
                    **params,
                )
                return _extract_content(response), _usage_dict(response)
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2 ** attempt)

        raise last_error or RuntimeError("OpenRouter request failed")


def _extract_content(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return json.dumps(content, ensure_ascii=False)


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "response_id": getattr(response, "id", None),
    }
