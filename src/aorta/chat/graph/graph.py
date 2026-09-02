"""LangGraph workflow with routing: Router → (Q&A | Action) paths.

Question path:  Router → Retrieve → Answer → End
Action path:    Router → Plan → Retrieve → Act ⇄ Critic → End
                                              └ Finalize → End (retries spent)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from aorta.chat.config import settings
from aorta.chat.graph.nodes import (
    act_node,
    answer_node,
    critic_node,
    finalize_node,
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
    """Accept, retry, or report an answer the critic would not accept.

    The third case is the point of the ``finalize`` branch. Sending exhaustion
    to ``END`` left the rejected ``AIMessage`` last in state, which is what
    ``invoke_agent`` reads its reply from -- so a budget spent entirely on
    rejections returned the last rejected answer as though the critic had
    passed it, and the verdict reached nobody.
    """
    if not state.get("critic_feedback"):
        return END
    if state.get("iteration", 0) < settings.max_retry_iterations:
        return "act"
    return "finalize"


def build_graph() -> StateGraph:
    """Construct and compile the agentic workflow graph.

    Flow:
        Router ──► question ──► Retrieve ──► Answer ──► End
        Router ──► action   ──► Plan ──► Retrieve ──► Act ⇄ Critic ──► End
                                                            └──► Finalize ──► End
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("act", act_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalize_node)
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
        {"act": "act", "finalize": "finalize", END: END},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


agent_graph = build_graph()
