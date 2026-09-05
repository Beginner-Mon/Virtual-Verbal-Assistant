from langchain_core.runnables import RunnableConfig

from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.nodes._persona_loader import get_ui_string
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.error_handler")


async def error_handler_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict:
    """Generate graceful error message from errors list.

    After Phase 6.9: writes final_answer directly (no longer routed through
    conversation node for styling — that node was deleted). The wording now
    comes from the selected character's `## UI Strings`, because this is the one
    message a user is guaranteed to read on a bad day and it was the only reply
    in the product that belonged to nobody.

    `config` is optional on purpose. LangGraph injects it either way, and a
    required second parameter would break every existing caller that passes only
    state — which is precisely why five grader tests are currently red with
    "grader_node() missing 1 required positional argument: 'config'".
    """
    errors = state.get("errors", [])
    critical = [e for e in errors if e.get("severity") == ErrorSeverity.CRITICAL]
    critical_count = len(critical)
    total_count = len(errors)

    # No hard-coded persona id here any more. This node runs when something has
    # already failed, and a broken persona is one of the things it may be
    # reporting — so it must not depend on one loading. `get_ui_string` swallows
    # a loader failure and falls back to neutral copy for the locale, which is
    # what keeps an error from turning into a second error.
    persona_id = ""
    locale = "en"
    if config:
        configurable = config.get("configurable", {})
        persona_id = configurable.get("persona_id") or ""
        locale = configurable.get("locale") or locale

    if critical:
        msg = get_ui_string(persona_id, "error_system", locale)
        logger.error("error_handler_invoked", extra={
            "node": "error_handler", "total_errors": total_count,
            "critical_errors": critical_count, "persona_id": persona_id,
        })
    else:
        msg = get_ui_string(persona_id, "error_partial", locale)
        logger.warning("error_handler_invoked", extra={
            "node": "error_handler", "total_errors": total_count,
            "persona_id": persona_id,
        })

    return {"final_answer": msg}
