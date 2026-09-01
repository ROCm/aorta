"""LangGraph workflow with routing: Router → (Q&A | Action) paths.

Question path:  Router → Retrieve → Answer → End
Action path:    Router → Plan → Retrieve → Act ⇄ Critic → End
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from aorta.chat.config import settings
from aorta.chat.graph.nodes import (
    act_node,
    answer_node,
    critic_node,
    plan_node,
    retrieve_node,
    router_node,
)
from aorta.chat.graph.state import AgentState


def route_after_router(state: AgentState) -> str:
    """Send 'action' queries through the Plan node, 'question' to Retrieve."""
    return "plan" if state.get("route") == "action" else "retrieve"


def route_after_retrieve(state: AgentState) -> str:
    """After retrieval, enter Act loop for actions or Answer for questions."""
    return "act" if state.get("route") == "action" else "answer"


def route_after_critic(state: AgentState) -> str:
    max_retries = settings.max_retry_iterations
    if state.get("critic_feedback") and state.get("iteration", 0) < max_retries:
        return "act"
    return END


def build_graph() -> StateGraph:
    """Construct and compile the agentic workflow graph.

    Flow:
        Router ──► question ──► Retrieve ──► Answer ──► End
        Router ──► action   ──► Plan ──► Retrieve ──► Act ⇄ Critic ──► End
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("act", act_node)
    graph.add_node("critic", critic_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"plan": "plan", "retrieve": "retrieve"},
    )

    graph.add_edge("plan", "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"act": "act", "answer": "answer"},
    )

    graph.add_edge("answer", END)
    graph.add_edge("act", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"act": "act", END: END},
    )

    return graph.compile()


agent_graph = build_graph()
