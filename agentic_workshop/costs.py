"""Token prices and the cost of an API call."""

from .config import CHAT_MODEL

# USD per 1M tokens. Check https://openai.com/api/pricing/ before using these for a real budget.
PRICES = {
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "cached_input": 0.02, "output": 0.0},
}


def usage_cost(usages, model: str = CHAT_MODEL) -> dict:
    """Tokens and cost (USD) of one usage object or a list of them.

    Accepts the `usage` of a Responses API call (input/output tokens, of which cached and reasoning
    tokens) and of an embeddings call (prompt tokens only).
    """
    if not isinstance(usages, (list, tuple)):
        usages = [usages]
    input_tokens = cached_tokens = output_tokens = reasoning_tokens = 0
    for u in usages:
        if hasattr(u, "input_tokens"):  # Responses API
            input_tokens += u.input_tokens
            output_tokens += u.output_tokens
            cached_tokens += getattr(u.input_tokens_details, "cached_tokens", 0) or 0
            reasoning_tokens += getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0
        else:  # embeddings
            input_tokens += u.prompt_tokens
    price = PRICES[model]
    cost = ((input_tokens - cached_tokens) * price["input"]
            + cached_tokens * price["cached_input"]
            + output_tokens * price["output"]) / 1e6
    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,  # includes the reasoning tokens
        "reasoning_tokens": reasoning_tokens,
        "cost_usd": cost,
    }


def print_cost(usages, model: str = CHAT_MODEL) -> None:
    c = usage_cost(usages, model)
    print(f"Input tokens:  {c['input_tokens']} (cached: {c['cached_tokens']})")
    print(f"Output tokens: {c['output_tokens']} (reasoning: {c['reasoning_tokens']})")
    print(f"Estimated cost: ${c['cost_usd']:.6f}")
