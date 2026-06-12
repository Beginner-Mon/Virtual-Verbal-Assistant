"""Context assembly — M.7 prompt caching layout (Tier 1).

Tier 1 layout (applies to ALL providers, always needed):

    [tools][system + user_memory facts][summary chunks (frozen)]   ← STATIC (prefix cache)
    ─────────────────────────────────────────────────────────────
    [recent raw][query]                                            ← DYNAMIC (each turn)

Principles (provider-independent):
  - Static first, dynamic last → prefix stays unchanged between turns
  - Summary chunks "frozen" on write (append-only) → prefix never invalidated
  - Cost per turn = size of recent raw (dynamic part) — NOT total context
  - Append-only: new summary chunk added at END of static block → cache hit

Tier 2 (provider-specific):
  - DeepSeek (current): auto-caches prefix. No cache_control needed.
  - Anthropic (future): add cache_control: ephemeral after last static block.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token (Vietnamese/English)."""
    return len(text) // 4 if text else 0


def estimate_context_tokens(
    system_prompt: str = "",
    user_facts: list[str] | None = None,
    summary_chunks: list[str] | None = None,
    recent_messages: list | None = None,
    query: str = "",
) -> dict:
    """Estimate token usage for context window per M.7 layout.

    Returns breakdown: {static_tokens, dynamic_tokens, total_tokens}
    """
    static = estimate_tokens(system_prompt)
    for f in (user_facts or []):
        static += estimate_tokens(f)
    for s in (summary_chunks or []):
        static += estimate_tokens(s)

    dynamic = 0
    for m in (recent_messages or []):
        if hasattr(m, "content"):
            dynamic += estimate_tokens(str(m.content))
    dynamic += estimate_tokens(query)

    return {
        "static_tokens": static,
        "dynamic_tokens": dynamic,
        "total_tokens": static + dynamic,
    }


def assemble_context_messages(
    system_prompt: str = "",
    user_facts: list[str] | None = None,
    summary_chunks: list[str] | None = None,
    recent_messages: list | None = None,
) -> list:
    """Assemble context messages in M.7 Tier 1 layout.

    Returns list of LangChain messages in correct order for LLM call.
    Static blocks first (cacheable), then dynamic recent messages.

    The caller adds the current HumanMessage(query) after these.
    """
    messages = []

    # STATIC block 1: system prompt + user facts
    static_parts = []
    if system_prompt:
        static_parts.append(system_prompt)
    if user_facts:
        facts_block = "[USER FACTS]\n" + "\n".join(f"- {f}" for f in user_facts)
        static_parts.append(facts_block)
    if static_parts:
        messages.append(SystemMessage(content="\n\n".join(static_parts)))

    # STATIC block 2: summary chunks (frozen, append-only)
    if summary_chunks:
        summaries = "[SESSION HISTORY]\n" + "\n---\n".join(summary_chunks)
        messages.append(SystemMessage(content=summaries))

    # DYNAMIC: recent raw messages
    if recent_messages:
        messages.extend(recent_messages)

    return messages
