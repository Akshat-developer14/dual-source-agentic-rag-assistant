"""LangGraph StateGraph workflow definition for Kara AWS Assistant.

Orchestrates multi-source retrieval, intent classification, and Corrective RAG (CRAG)
feedback loops with dynamic self-correction and web search fallback.
"""

from typing import Literal
from langgraph.graph import END, START, StateGraph

from src.nodes import (
    contextualize_node,
    grade_context_node,
    retriever_node,
    rewrite_query_node,
    router_node,
    synthesizer_node,
    web_search_node,
)
from src.state import AgentState

# Maximum query reformulation attempts before falling back to external web search
MAX_RETRIEVAL_RETRIES = 1


def decide_route(state: AgentState) -> Literal["retriever_node", "web_search_node", "__end__"]:
    """Determines initial execution path based on the router's intent classification."""
    route = state.get("route")
    if route == "web":
        return "web_search_node"
    elif route in ["chitchat", "unrelated"]:
        return "__end__"
    return "retriever_node"


def decide_post_grade_route(
    state: AgentState,
) -> Literal["synthesizer_node", "rewrite_query_node", "web_search_node"]:
    """Evaluates context sufficiency to route to synthesis, query rewrite, or web search fallback."""
    is_sufficient = state.get("is_sufficient", False)
    retries = state.get("retry_count", 0)

    # 1. Context fully satisfies user query -> proceed to synthesis
    if is_sufficient:
        return "synthesizer_node"

    # 2. Context insufficient but retry budget remains -> reformulate query
    if retries < MAX_RETRIEVAL_RETRIES:
        return "rewrite_query_node"

    # 3. Context insufficient and retries exhausted -> fallback to live web search
    return "web_search_node"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------
builder = StateGraph(AgentState)

# Node registration
builder.add_node("contextualize_node", contextualize_node)
builder.add_node("router_node", router_node)
builder.add_node("retriever_node", retriever_node)
builder.add_node("grade_context_node", grade_context_node)
builder.add_node("rewrite_query_node", rewrite_query_node)
builder.add_node("web_search_node", web_search_node)
builder.add_node("synthesizer_node", synthesizer_node)

# Entry and intent routing edges
builder.add_edge(START, "contextualize_node")
builder.add_edge("contextualize_node", "router_node")

builder.add_conditional_edges(
    "router_node",
    decide_route,
    {
        "retriever_node": "retriever_node",
        "web_search_node": "web_search_node",
        END: END,
    },
)

# Corrective RAG (CRAG) evaluation and retry loop
builder.add_edge("retriever_node", "grade_context_node")

builder.add_conditional_edges(
    "grade_context_node",
    decide_post_grade_route,
    {
        "synthesizer_node": "synthesizer_node",
        "rewrite_query_node": "rewrite_query_node",
        "web_search_node": "web_search_node",
    },
)

builder.add_edge("rewrite_query_node", "retriever_node")

# Web search and synthesis convergence
builder.add_edge("web_search_node", "synthesizer_node")
builder.add_edge("synthesizer_node", END)

# Compile into executable graph
agent_graph = builder.compile()