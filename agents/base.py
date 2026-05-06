from typing import Generator

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import cfg

_client = OpenAI(api_key=cfg.LLM_API_KEY, base_url=cfg.LLM_BASE_URL)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(
    system: str,
    user: str,
    model: str = cfg.LLM_MODEL,
    max_tokens: int = cfg.MAX_TOKENS,
) -> str:
    response = _client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        inner = text.split("\n", 1)[-1].strip()
        if inner.startswith("<"):
            text = inner
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
    return text.strip()


def call_llm_stream(
    system: str,
    user: str,
    model: str = cfg.LLM_MODEL,
    max_tokens: int = cfg.MAX_TOKENS,
) -> Generator[str, None, None]:
    stream = _client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        stream=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def rag_block(context: str) -> str:
    return f"<retrieved_context>\n{context}\n</retrieved_context>"


def prior_outputs_block(outputs: dict[str, str]) -> str:
    parts = []
    for name, content in outputs.items():
        parts.append(f"<{name}_output>\n{content}\n</{name}_output>")
    return "\n\n".join(parts)
