from collections.abc import Callable

from climate_rag.config import Settings

# (system_prompt, user_prompt) -> completion. A plain callable keeps agents testable
# with a fake and decoupled from any particular SDK.
LLM = Callable[[str, str], str]


def make_chat(
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> LLM:
    """Chat completion against any OpenAI-compatible endpoint."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    return complete


def make_llm(settings: Settings) -> LLM:
    """The answer-generating model (Ollama by default)."""
    return make_chat(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        settings.llm_temperature,
        settings.llm_max_tokens,
        settings.llm_timeout,
    )


def make_expander(settings: Settings) -> LLM:
    """The generator model at temperature 0, so query expansions are repeatable."""
    return make_chat(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        0.0,
        200,
        settings.llm_timeout,
    )


def judge_model_name(settings: Settings) -> str:
    return settings.judge_model or settings.llm_model


def make_judge(settings: Settings) -> LLM:
    """The eval judge: deterministic, with room for a claim-by-claim JSON verdict."""
    return make_chat(
        settings.judge_base_url or settings.llm_base_url,
        settings.judge_api_key or settings.llm_api_key,
        judge_model_name(settings),
        0.0,
        settings.judge_max_tokens,
        settings.llm_timeout,
    )
