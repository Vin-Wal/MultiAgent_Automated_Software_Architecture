"""
Shared LLM client and helper functions used by all agents.

Provides a retry-wrapped blocking call, a streaming variant, and two
utility functions for building structured prompt blocks.
"""
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
    """
    Call the LLM and return the full response as a string.

    Retries up to 3 times with exponential back-off on transient errors.
    Strips Markdown code-fence wrappers if the model adds them around XML.

    Args:
        system: System prompt that sets the agent role and output format.
        user: User message (RAG context + task description).
        model: Model identifier, defaults to ``cfg.LLM_MODEL``.
        max_tokens: Maximum tokens to generate.

    Returns:
        The model's response text, stripped of leading/trailing whitespace.
    """
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
    """
    Stream the LLM response token-by-token.

    Yields each non-empty delta from the chat completion stream.
    Intended for the Streamlit UI where live output improves perceived latency.

    Args:
        system: System prompt.
        user: User message.
        model: Model identifier.
        max_tokens: Maximum tokens to generate.

    Yields:
        Successive string deltas from the model.
    """
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
    """Wrap retrieved context in an XML tag for the LLM prompt."""
    return f"<retrieved_context>\n{context}\n</retrieved_context>"


def prior_outputs_block(outputs: dict[str, str]) -> str:
    """
    Wrap prior agent outputs in named XML tags.

    Args:
        outputs: Mapping of output name to content string.

    Returns:
        A single string with each output wrapped in ``<name_output>`` tags.
    """
    parts = []
    for name, content in outputs.items():
        parts.append(f"<{name}_output>\n{content}\n</{name}_output>")
    return "\n\n".join(parts)
