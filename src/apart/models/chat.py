from __future__ import annotations

from typing import Any


def chat_messages(prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def render_generation_prompt(
    tokenizer: Any,
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    return tokenizer.apply_chat_template(
        chat_messages(prompt, system_prompt),
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_generation_prompt(
    tokenizer: Any,
    prompt: str,
    system_prompt: str | None = None,
) -> list[int]:
    rendered = render_generation_prompt(tokenizer, prompt, system_prompt)
    encoded = tokenizer(rendered, add_special_tokens=False)
    return [int(token) for token in encoded["input_ids"]]
