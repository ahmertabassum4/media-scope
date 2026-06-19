from experiment_bootstrap import activate_project_root

activate_project_root()

import os
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI


class OpenRouterLLM:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing in environment or .env")

        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
        )

    def openr_llm(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        params: dict[str, Any] = {
            **kwargs,
            "model": model,
            "messages": messages,
        }

        optional_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "response_format": response_format,
            "extra_body": extra_body,
            "timeout": timeout,
        }

        params.update({key: value for key, value in optional_params.items() if value is not None})

        return self.client.chat.completions.create(**params)
