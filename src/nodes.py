"""Graph node implementations for Kara AWS Assistant.

Each node represents an execution step within the LangGraph StateGraph,
encapsulating LLM invocations, vectorstore retrievals, context grading,
query reformulation, external web search, token tracking, and grounded synthesis.
"""

import hashlib
import json
import logging
import re
import warnings
from typing import Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.prompts import (
    ROUTER_SYSTEM_PROMPT,
    CONTEXTUALIZE_SYSTEM_PROMPT,
    GRADER_SYSTEM_PROMPT,
    REWRITER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)
from src.state import AgentState
from src.tools import get_internal_retriever, search_web

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# LLM Initialization & Rate-Limit Resilience
# Workloads are distributed across high-performance, instruction-aligned Groq models:
#   - openai/gpt-oss-20b   (Fast, precise; contextualization, routing, rewrites)
#   - openai/gpt-oss-120b  (Deep reasoning; grading, synthesis, fallback)
# ---------------------------------------------------------------------------

# Router LLM: Intent classification with structured JSON output
_router_primary = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    max_retries=3,
    model_kwargs={"response_format": {"type": "json_object"}},
)
_router_fallback = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_retries=3,
    model_kwargs={"response_format": {"type": "json_object"}},
)
ROUTER_LLM = _router_primary.with_fallbacks([_router_fallback])

# Contextualizer LLM: Follow-up query de-referencing & normalization
_ctx_primary = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    max_retries=3,
)
_ctx_fallback = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_retries=3,
)
CONTEXTUALIZE_LLM = _ctx_primary.with_fallbacks([_ctx_fallback])

# Grader LLM: Context sufficiency evaluation & dynamic k selection
_grader_primary = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_retries=3,
    model_kwargs={"response_format": {"type": "json_object"}},
)
_grader_fallback = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    max_retries=3,
    model_kwargs={"response_format": {"type": "json_object"}},
)
GRADER_LLM = _grader_primary.with_fallbacks([_grader_fallback])

# Rewriter LLM: Search query reformulation based on grader feedback
_rewriter_primary = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_retries=3,
)
_rewriter_fallback = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_retries=3,
)
REWRITER_LLM = _rewriter_primary.with_fallbacks([_rewriter_fallback])

# Synthesizer LLM: Grounded answer generation with cited sources
_synth_primary = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_retries=3,
    model_kwargs={"response_format": {"type": "json_object"}},
)
_synth_fallback = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_retries=3,
    model_kwargs={"response_format": {"type": "json_object"}},
)
SYNTHESIZER_LLM = _synth_primary.with_fallbacks([_synth_fallback])


def _extract_tokens(response: Any) -> tuple[int, int, int]:
    """Extracts (prompt_tokens, completion_tokens, total_tokens) from an LLM response."""
    usage = getattr(response, "usage_metadata", None) or {}
    if not usage:
        resp_meta = getattr(response, "response_metadata", {}) or {}
        usage = resp_meta.get("token_usage") or {}
    p = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    c = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    t = usage.get("total_tokens") or (p + c)
    return p, c, t


def _strip_think_tags(text: str) -> str:
    """Strips <think>...</think> reasoning traces emitted by reasoning models."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        match = re.search(r"(?:Output|Rewritten|Result):\s*([\s\S]+)$", cleaned, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        cleaned = re.sub(r"^<think>\s*", "", cleaned).strip()
    return cleaned


def _extract_json(text: str) -> dict:
    """Extracts and parses JSON from raw LLM output, resilient to thinking tags & fences."""
    cleaned = _strip_think_tags(text)

    # Extract content within markdown code fences if present
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    # Attempt direct JSON deserialization
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Fallback: Extract outermost JSON object { ... }
    match = re.search(r"(\{[\s\S]*\})", cleaned)
    if match:
        return json.loads(match.group(1))

    raise ValueError(f"No valid JSON found in model output: {text[:120]}...")


def contextualize_node(state: AgentState) -> dict:
    """Reformulates follow-up user questions into self-contained standalone queries."""
    history = (state.get("chat_history") or [])[-8:]
    question = (state.get("question") or "").strip()

    p_acc = state.get("prompt_tokens") or 0
    c_acc = state.get("completion_tokens") or 0
    t_acc = state.get("total_tokens") or 0

    if not history:
        return {
            "standalone_question": question,
            "prompt_tokens": p_acc,
            "completion_tokens": c_acc,
            "total_tokens": t_acc,
        }

    history_text = ""
    for msg in history:
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    prompt = f"Chat History:\n{history_text}\nLatest Message: {question}\n\nRewritten:"

    try:
        response = CONTEXTUALIZE_LLM.invoke([
            SystemMessage(content=CONTEXTUALIZE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        p, c, t = _extract_tokens(response)
        content = response.content if isinstance(response.content, str) else str(response.content)
        rewritten = _strip_think_tags(content).strip('"\' ')
        return {
            "standalone_question": rewritten if rewritten else question,
            "prompt_tokens": p_acc + p,
            "completion_tokens": c_acc + c,
            "total_tokens": t_acc + t,
        }
    except Exception:
        logger.warning("Contextualization invocation failed; falling back to raw question.")
        return {
            "standalone_question": question,
            "prompt_tokens": p_acc,
            "completion_tokens": c_acc,
            "total_tokens": t_acc,
        }


def router_node(state: AgentState) -> dict[str, Any]:
    """Classifies user query into 'internal', 'web', 'chitchat', or 'unrelated' with CoT reasoning."""
    query_to_route = _strip_think_tags(
        state.get("standalone_question") or state.get("question") or ""
    ).strip()

    p_acc = state.get("prompt_tokens") or 0
    c_acc = state.get("completion_tokens") or 0
    t_acc = state.get("total_tokens") or 0

    if not query_to_route:
        return {
            "thought": "No query text available to route.",
            "route": "chitchat",
            "direct_response": "Hi! What AWS or cloud question can I help you with?",
            "answer": "Hi! What AWS or cloud question can I help you with?",
            "retry_count": 0,
            "k": 4,
            "prompt_tokens": p_acc,
            "completion_tokens": c_acc,
            "total_tokens": t_acc,
        }

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"User Query: {query_to_route}"),
    ]

    thought = "Routing based on user query intent."
    route = "internal"
    direct_response = ""

    try:
        response = ROUTER_LLM.invoke(messages)
        p, c, t = _extract_tokens(response)
        p_acc += p
        c_acc += c
        t_acc += t

        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        data = _extract_json(content)
        thought = data.get("thought", thought)
        parsed_route = str(data.get("route", "internal")).strip().lower()
        if parsed_route in ["internal", "web", "chitchat", "unrelated"]:
            route = parsed_route
        direct_response = str(data.get("direct_response", "") or "").strip()
    except Exception:
        logger.warning("Router classification failed; falling back to internal route.")
        thought = "Fallback: routing to internal knowledge base."
        route = "internal"
        direct_response = ""

    if route in ("internal", "web"):
        direct_response = ""
    elif not direct_response:
        direct_response = (
            "Happy to help — what AWS or cloud question can I answer for you?"
            if route == "chitchat"
            else "That's outside what I can help with — I'm focused on AWS and cloud "
                 "architecture questions. Let me know if you have one of those!"
        )

    return {
        "thought": thought,
        "route": route,
        "direct_response": direct_response,
        "answer": direct_response,
        "retry_count": 0,
        "k": 4,
        "prompt_tokens": p_acc,
        "completion_tokens": c_acc,
        "total_tokens": t_acc,
    }


def retriever_node(state: AgentState) -> dict:
    """Retrieves document chunks from the local FAISS index with MD5-based deduplication."""
    current_k = state.get("k", 4)
    if current_k is None:
        current_k = 4

    query_to_search = _strip_think_tags(
        state.get("revised_question")
        or state.get("standalone_question")
        or state.get("question", "")
    ).strip()

    logger.info("Executing retrieval for query: %r with density k=%d", query_to_search, current_k)

    retriever = get_internal_retriever(k=current_k)
    docs = retriever.invoke(query_to_search)

    if not docs:
        logger.warning("No document chunks returned for query: %r", query_to_search)
        return {"context": "", "sources": []}

    seen_hashes = set()
    context_chunks = []
    sources = []

    for doc in docs:
        page_content = doc.page_content.strip()
        chunk_hash = hashlib.md5(page_content.encode("utf-8")).hexdigest()
        if chunk_hash in seen_hashes:
            continue
        seen_hashes.add(chunk_hash)

        page_num = doc.metadata.get("page", doc.metadata.get("page_number", "N/A"))
        source_name = doc.metadata.get("source", "AWS Documentation")

        clean_source_name = (
            source_name.replace("data/", "")
            .replace(".pdf", "")
            .replace("_", " ")
            .title()
        )
        source_label = f"{clean_source_name} (Page {page_num})"

        context_chunks.append(f"[Source: {source_label}]\n{page_content}")
        if source_label not in sources:
            sources.append(source_label)

    aggregated_context = "\n\n---\n\n".join(context_chunks)
    logger.info("Retrieved %d distinct chunks across %d unique sources.", len(context_chunks), len(sources))

    return {"context": aggregated_context, "sources": sources}


def grade_context_node(state: AgentState) -> dict:
    """Evaluates context completeness and dynamically adjusts retrieval density k."""
    current_k = state.get("k", 4)
    if current_k is None:
        current_k = 4

    p_acc = state.get("prompt_tokens") or 0
    c_acc = state.get("completion_tokens") or 0
    t_acc = state.get("total_tokens") or 0

    question_used = _strip_think_tags(
        state.get("revised_question")
        or state.get("standalone_question")
        or state.get("question", "")
    )
    context = state.get("context") or ""

    if not context.strip():
        logger.info("Context is empty; marking insufficient to trigger reformulation/fallback.")
        return {
            "is_sufficient": False,
            "k": min(10, current_k + 2),
            "grader_thought": "No context chunks were retrieved.",
            "prompt_tokens": p_acc,
            "completion_tokens": c_acc,
            "total_tokens": t_acc,
        }

    prompt = f"Retrieved Context:\n{context}\n\nUser Question: {question_used}"

    try:
        response = GRADER_LLM.invoke(
            [SystemMessage(content=GRADER_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        p, c, t = _extract_tokens(response)
        p_acc += p
        c_acc += c
        t_acc += t

        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        data = _extract_json(content)
    except Exception:
        logger.warning("Grader evaluation failed; defaulting to sufficient context.")
        return {
            "is_sufficient": True,
            "k": current_k,
            "grader_thought": "",
            "prompt_tokens": p_acc,
            "completion_tokens": c_acc,
            "total_tokens": t_acc,
        }

    is_sufficient = bool(data.get("is_sufficient", True))

    try:
        raw_k = int(data.get("recommended_k", current_k))
    except (TypeError, ValueError):
        raw_k = current_k
    recommended_k = max(4, min(10, raw_k))
    if not is_sufficient:
        recommended_k = max(recommended_k, current_k)

    grader_thought = str(data.get("thought", "")).strip()

    return {
        "is_sufficient": is_sufficient,
        "k": recommended_k,
        "grader_thought": grader_thought,
        "prompt_tokens": p_acc,
        "completion_tokens": c_acc,
        "total_tokens": t_acc,
    }


def rewrite_query_node(state: AgentState) -> dict:
    """Reformulates the search query incorporating grader feedback and increments retry count."""
    current_retries = state.get("retry_count", 0)
    p_acc = state.get("prompt_tokens") or 0
    c_acc = state.get("completion_tokens") or 0
    t_acc = state.get("total_tokens") or 0

    question_used = _strip_think_tags(
        state.get("revised_question")
        or state.get("standalone_question")
        or state.get("question", "")
    )
    grader_note = state.get("grader_thought") or "Context was insufficient or off-topic."

    prompt = f"Original Question: {question_used}\nGrader Note: {grader_note}"

    try:
        response = REWRITER_LLM.invoke(
            [
                SystemMessage(content=REWRITER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        p, c, t = _extract_tokens(response)
        p_acc += p
        c_acc += c
        t_acc += t

        content = response.content if isinstance(response.content, str) else str(response.content)
        revised_query = _strip_think_tags(content).strip('"\' ') or question_used
    except Exception:
        logger.warning("Query rewrite failed; retaining previous query string.")
        revised_query = question_used

    return {
        "revised_question": revised_query,
        "retry_count": current_retries + 1,
        "prompt_tokens": p_acc,
        "completion_tokens": c_acc,
        "total_tokens": t_acc,
    }


def web_search_node(state: AgentState) -> dict:
    """Fetches real-time web context using Tavily Search API with retry backoff."""
    query_to_search = _strip_think_tags(
        state.get("revised_question")
        or state.get("standalone_question")
        or state.get("question", "")
    )
    try:
        context, sources = search_web(query_to_search)
    except Exception:
        logger.warning("Web search execution failed for query: %r", query_to_search)
        context = "The web search failed and returned no results."
        sources = []
    return {"context": context, "sources": sources}


def synthesizer_node(state: AgentState) -> dict:
    """Synthesizes the final grounded answer strictly from the retrieved context."""
    p_acc = state.get("prompt_tokens") or 0
    c_acc = state.get("completion_tokens") or 0
    t_acc = state.get("total_tokens") or 0

    question_used = _strip_think_tags(
        state.get("standalone_question") or state.get("question", "")
    )
    context = state.get("context") or "No context available."
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question_used}"

    try:
        response = SYNTHESIZER_LLM.invoke([
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        p, c, t = _extract_tokens(response)
        p_acc += p
        c_acc += c
        t_acc += t

        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        data = _extract_json(content)

        answer = str(data.get("answer", "")).strip()
        if not answer:
            answer = "I wasn't able to generate a complete answer from the retrieved context."

        sources = data.get("sources")
        if not isinstance(sources, list):
            sources = state.get("sources") or []

        return {
            "answer": answer,
            "sources": sources,
            "prompt_tokens": p_acc,
            "completion_tokens": c_acc,
            "total_tokens": t_acc,
        }
    except Exception:
        logger.warning("Synthesizer response generation failed.")
        return {
            "answer": "Sorry, I ran into an issue generating a response. Could you try rephrasing your question?",
            "sources": [],
            "prompt_tokens": p_acc,
            "completion_tokens": c_acc,
            "total_tokens": t_acc,
        }
