from langgraph_agents.state import AgentState, ErrorSeverity


async def error_handler_node(state: AgentState) -> dict:
    """Generate graceful error message from errors list."""
    errors = state.get("errors", [])
    critical = [e for e in errors if e.get("severity") == ErrorSeverity.CRITICAL]

    if critical:
        msg = "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
    else:
        msg = "Đã có lỗi nhỏ, nhưng tôi vẫn cố gắng trả lời."

    return {"reasoning_output": msg}
