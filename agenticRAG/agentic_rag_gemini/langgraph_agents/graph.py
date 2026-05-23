from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from langgraph_agents.state import AgentState
from langgraph_agents.nodes.memory import memory_node
from langgraph_agents.nodes.planner import planner_node
from langgraph_agents.nodes.retriever_agent import retriever_agent_node, RETRIEVER_TOOLS
from langgraph_agents.nodes.synthesizer import synthesizer_node
from langgraph_agents.nodes.grader import grader_node
from langgraph_agents.nodes.conversation import conversation_node
from langgraph_agents.nodes.error_handler import error_handler_node
from langgraph_agents.routing import (
    route_after_memory,
    route_after_planner,
    route_after_retriever,
    route_after_grader,
    check_errors,
)


_RECURSION_LIMIT = 15  # planner→retriever⇄tools→synth→grader (retry once) fits comfortably


def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("memory", memory_node)
    g.add_node("planner", planner_node)
    g.add_node("retriever_agent", retriever_agent_node)
    g.add_node("tools", ToolNode(RETRIEVER_TOOLS))
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("grader", grader_node)
    g.add_node("conversation", conversation_node)
    g.add_node("error_handler", error_handler_node)

    g.add_edge(START, "memory")

    g.add_conditional_edges("memory", route_after_memory, {
        "planner": "planner",
        "error_handler": "error_handler",
    })

    g.add_conditional_edges("planner", route_after_planner, {
        "conversation": "conversation",
        "retriever_agent": "retriever_agent",
        "error_handler": "error_handler",
    })

    # retriever_agent: check errors first, then tool_calls vs done
    g.add_conditional_edges("retriever_agent", route_after_retriever, {
        "tools": "tools",
        "synthesizer": "synthesizer",
        "error_handler": "error_handler",
    })
    g.add_edge("tools", "retriever_agent")

    g.add_conditional_edges("synthesizer", check_errors, {
        "continue": "grader",
        "error_handler": "error_handler",
    })

    g.add_conditional_edges("grader", route_after_grader, {
        "retriever_agent": "retriever_agent",
        "conversation": "conversation",
    })

    g.add_edge("conversation", END)
    g.add_edge("error_handler", "conversation")

    return g.compile().with_config(recursion_limit=_RECURSION_LIMIT)
