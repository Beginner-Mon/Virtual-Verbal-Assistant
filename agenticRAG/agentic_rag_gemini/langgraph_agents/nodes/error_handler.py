from langgraph_agents.state import AgentState, ErrorSeverity
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.error_handler")


async def error_handler_node(state: AgentState) -> dict:
    """Generate graceful error message from errors list.

    After Phase 6.9: writes final_answer directly (no longer routed through
    conversation node for styling — that node was deleted).
    """
    errors = state.get("errors", [])
    critical = [e for e in errors if e.get("severity") == ErrorSeverity.CRITICAL]
    critical_count = len(critical)
    total_count = len(errors)

    if critical:
        msg = "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
        logger.error("error_handler_invoked", extra={
            "node": "error_handler", "total_errors": total_count,
            "critical_errors": critical_count,
        })
    else:
        msg = "Đã có lỗi nhỏ, nhưng tôi vẫn cố gắng trả lời."
        logger.warning("error_handler_invoked", extra={
            "node": "error_handler", "total_errors": total_count,
        })

    return {"final_answer": msg}
