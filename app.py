"""Streamlit web interface for Kara — AWS Cloud Architecture Assistant.

Provides interactive chat, live node-level execution streaming, source citations,
and dynamic metadata inspection for LangGraph execution traces.
"""

import logging
import streamlit as st

from src.graph import agent_graph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kara — AWS Cloud Assistant",
    page_icon="☁️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# UI Theme & Custom Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Top Accent Gradient Bar */
    .accent-bar {
        height: 3px;
        background: linear-gradient(90deg, #FF9900 0%, #f59e0b 50%, #FF9900 100%);
        border-radius: 2px;
        margin-bottom: 1.2rem;
    }

    /* Main App Header */
    .kara-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.1rem;
    }
    .kara-header h1 {
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #FF9900 0%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .kara-subheading {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-bottom: 1.4rem;
        letter-spacing: 0.03em;
    }

    /* Preset Prompt Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(255, 153, 0, 0.4);
        color: #FF9900;
        background-color: rgba(255, 153, 0, 0.04);
        font-size: 0.82rem;
        padding: 0.5rem 0.8rem;
        transition: all 0.18s ease;
        text-align: left;
        white-space: normal;
        height: auto !important;
        line-height: 1.4;
    }
    .stButton > button:hover {
        background-color: #FF9900;
        color: white;
        border-color: #FF9900;
        box-shadow: 0 2px 12px rgba(255, 153, 0, 0.35);
        transform: translateY(-1px);
    }

    /* Sidebar Dark Theme Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stButton > button {
        border-color: rgba(255,153,0,0.5) !important;
        color: #FF9900 !important;
        background: rgba(255,153,0,0.07) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #FF9900 !important;
        color: white !important;
    }

    /* Technology Stack Pills */
    .badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.4rem 0 0.8rem 0; }
    .tech-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        background: rgba(255,153,0,0.12);
        color: #FF9900 !important;
        border: 1px solid rgba(255,153,0,0.3);
    }

    /* Route Legend Layout */
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.8rem;
    }
    .legend-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* Route Badge in Execution Trace */
    .route-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        color: white;
        letter-spacing: 0.02em;
    }

    /* Execution Trace Expander */
    [data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid rgba(150, 150, 150, 0.15);
        background: rgba(0,0,0,0.02);
    }

    /* Citation Source Cards */
    .source-card {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 8px;
        border: 1px solid rgba(150,150,150,0.15);
        margin-bottom: 5px;
        font-size: 0.82rem;
        background: rgba(0,0,0,0.02);
        text-decoration: none;
    }
    .source-card:hover { border-color: #FF9900; }

    /* Performance Metric Chips */
    .metric-chip {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        padding: 8px 16px;
        border-radius: 10px;
        border: 1px solid rgba(150,150,150,0.15);
        background: rgba(0,0,0,0.02);
        min-width: 80px;
        text-align: center;
    }
    .metric-chip .val { font-size: 1.1rem; font-weight: 700; }
    .metric-chip .lbl { font-size: 0.68rem; color: #9ca3af; margin-top: 2px; }

    /* Suggestion Heading */
    .try-heading {
        font-size: 0.9rem;
        font-weight: 600;
        color: #9ca3af;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Visual Constants & Route Specifications
# ---------------------------------------------------------------------------
ROUTE_STYLES = {
    "internal":  ("📘 Internal Docs",  "#0972d3"),
    "web":       ("🌐 Live Web",       "#7c3aed"),
    "chitchat":  ("💬 Chitchat",       "#059669"),
    "unrelated": ("🚫 Out of Scope",   "#dc2626"),
}

ROUTE_DESCRIPTIONS = {
    "internal":  "Answered from local AWS docs",
    "web":       "Answered via live web search",
    "chitchat":  "Conversational reply, no retrieval",
    "unrelated": "Out-of-scope — deflected politely",
}

EXAMPLE_PROMPTS = [
    "What are the 6 pillars of the AWS Well-Architected Framework?",
    "Is AWS us-east-1 having any issues right now?",
    "Does AWS offer inference hosting for open-source LLMs?",
]

# Sliding conversational history window (bounded payload)
HISTORY_WINDOW = 16

# Status descriptions rendered during live node streaming
NODE_STATUS = {
    "contextualize_node": "🔤 Contextualizing query...",
    "router_node":        "🧭 Analyzing intent...",
    "retriever_node":     "📂 Searching internal docs...",
    "grade_context_node": "🔬 Grading retrieved context...",
    "rewrite_query_node": "✏️  Rewriting query for better results...",
    "web_search_node":    "🌐 Searching the live web...",
    "synthesizer_node":   "🧠 Synthesizing answer...",
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def route_badge(route: str) -> str:
    """Renders a styled HTML badge pill for the selected route."""
    label, color = ROUTE_STYLES.get(route, ("❓ Unknown", "#6b7280"))
    return f'<span class="route-badge" style="background:{color};">{label}</span>'


def is_url(value: str) -> bool:
    """Checks whether a given string is a valid web URL."""
    return isinstance(value, str) and value.startswith("http")


def render_source(src: str):
    """Renders a formatted citation card for a web URL or document page."""
    if is_url(src):
        domain = src.split("/")[2] if "/" in src else src
        st.markdown(
            f'<a class="source-card" href="{src}" target="_blank">'
            f'🔗 <span style="color:#0972d3;font-weight:500;">{domain}</span>'
            f'<span style="color:#9ca3af;font-size:0.75rem;margin-left:4px;">{src[:60]}{"…" if len(src)>60 else ""}</span>'
            f'</a>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="source-card">📄 {src}</div>',
            unsafe_allow_html=True,
        )


def render_trace(meta: dict):
    """Renders the expandable execution trace and decision metrics."""
    with st.expander("🔍 Agent Reasoning & Trace", expanded=False):
        route = meta.get("route")

        # Contextualized query inspection
        if (
            meta.get("standalone_question")
            and meta.get("standalone_question") != meta.get("raw_question")
        ):
            st.markdown(
                f"**Contextualized query:** `{meta.get('standalone_question')}`"
            )
            st.divider()

        # Route badge and router thought
        st.markdown(
            f"**Route:** {route_badge(route)}", unsafe_allow_html=True
        )
        if meta.get("thought"):
            st.caption(f"💭 {meta.get('thought')}")

        # Metrics for internal Corrective RAG path
        if route == "internal":
            st.markdown("")
            sufficient = meta.get("is_sufficient")
            retries = meta.get("retry_count", 0)
            k = meta.get("k", "—")
            suf_val = "✅ Yes" if sufficient else ("❌ No" if sufficient is not None else "—")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="metric-chip"><span class="val">{suf_val}</span>'
                    f'<span class="lbl">Sufficient</span></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="metric-chip"><span class="val">{retries}</span>'
                    f'<span class="lbl">Retries</span></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="metric-chip"><span class="val">{k}</span>'
                    f'<span class="lbl">Chunks (k)</span></div>',
                    unsafe_allow_html=True,
                )

        elif route == "web":
            st.caption("🌐 Live web search was used — results reflect current operational data.")
        elif route == "chitchat":
            st.caption("💬 Conversational reply handled at the routing layer — no retrieval needed.")
        elif route == "unrelated":
            st.caption("🚫 Off-topic query detected and deflected at the routing layer.")

        # Citation sources
        sources = meta.get("sources")
        if sources:
            st.markdown("")
            st.markdown("**Sources cited:**")
            for src in sources:
                render_source(src)


# ---------------------------------------------------------------------------
# Sidebar Interface
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:1rem 0 0.5rem;">'
        '<span style="font-size:2.2rem;">☁️</span><br>'
        '<span style="font-size:1.2rem;font-weight:700;color:#FF9900;">Kara</span><br>'
        '<span style="font-size:0.75rem;color:#94a3b8;">AWS Cloud Architecture Assistant</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**How it works**")
    st.markdown(
        "<span style='font-size:0.82rem;color:#cbd5e1;'>"
        "Kara uses a self-correcting RAG pipeline: it routes your question, "
        "retrieves from local AWS docs or the live web, grades context quality, "
        "and retries with a rewritten query if needed."
        "</span>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("**Powered by**")
    st.markdown(
        '<div class="badge-row">'
        '<span class="tech-badge">LangGraph</span>'
        '<span class="tech-badge">Groq</span>'
        '<span class="tech-badge">FAISS</span>'
        '<span class="tech-badge">Tavily</span>'
        '<span class="tech-badge">Streamlit</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("**Route Legend**")
    for route, (label, color) in ROUTE_STYLES.items():
        desc = ROUTE_DESCRIPTIONS[route]
        st.markdown(
            f'<div class="legend-item">'
            f'<span class="legend-dot" style="background:{color};"></span>'
            f'<span style="font-size:0.78rem;"><b>{label}</b> — {desc}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("No login, nothing saved — refreshing starts fresh.")

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="kara-header"><h1>☁️ Kara</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="kara-subheading">'
    "Self-Reflective Agentic RAG &nbsp;·&nbsp; AWS Well-Architected Framework "
    "&nbsp;·&nbsp; Live Web Search &nbsp;·&nbsp; Powered by LangGraph &amp; Groq"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous conversational turns
for msg in st.session_state.messages:
    avatar = "☁️" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "metadata" in msg:
            render_trace(msg["metadata"])

# ---------------------------------------------------------------------------
# Chat Input & Starter Prompts
# ---------------------------------------------------------------------------
prompt = st.chat_input(
    "Ask about AWS architecture, EC2 pricing, outages, or cloud best practices..."
)

# Render starter prompt pills for initial session state
if not st.session_state.messages and not prompt:
    st.markdown(
        '<p class="try-heading">✨ Try asking</p>', unsafe_allow_html=True
    )
    cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, example in zip(cols, EXAMPLE_PROMPTS):
        if col.button(example, use_container_width=True):
            prompt = example

# ---------------------------------------------------------------------------
# Interaction Handler with Node-Level Streaming
# ---------------------------------------------------------------------------
if prompt:
    # Build history context bounded by HISTORY_WINDOW
    recent_history = st.session_state.messages[-HISTORY_WINDOW:]
    clean_history = [
        {"role": m["role"], "content": m["content"]}
        for m in recent_history
        if m.get("role") in ("user", "assistant")
    ]

    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Execute graph with real-time status streaming
    with st.chat_message("assistant", avatar="☁️"):
        with st.status("Kara is thinking...", expanded=True) as status_box:
            try:
                full_state: dict = {
                    "question": prompt,
                    "chat_history": clean_history,
                }

                for chunk in agent_graph.stream(
                    full_state,
                    stream_mode="updates",
                ):
                    node_name, updates = next(iter(chunk.items()))

                    if node_name.startswith("__"):
                        continue

                    if isinstance(updates, dict):
                        full_state.update(updates)

                    base_label = NODE_STATUS.get(node_name, f"⚙️ {node_name}...")

                    if node_name == "router_node":
                        route = full_state.get("route", "")
                        _, color = ROUTE_STYLES.get(route, ("", "#6b7280"))
                        st.write(
                            f"🧭 **Router decided:** "
                            f'<span class="route-badge" style="background:{color};font-size:0.75rem;">'
                            f'{ROUTE_STYLES.get(route, (route,""))[0]}</span>',
                            unsafe_allow_html=True,
                        )

                    elif node_name == "grade_context_node":
                        sufficient = full_state.get("is_sufficient")
                        retries = full_state.get("retry_count", 0)
                        if sufficient:
                            st.write("🔬 **Context graded:** sufficient — moving to synthesis")
                        else:
                            st.write(
                                f"🔬 **Context graded:** insufficient "
                                f"(retry #{retries + 1}) — rewriting query..."
                            )

                    elif node_name == "rewrite_query_node":
                        revised = full_state.get("revised_question", "")
                        st.write(f"✏️ **Rewritten query:** *{revised}*")

                    else:
                        st.write(base_label)

                result = full_state
                status_box.update(label="Done ✓", state="complete", expanded=False)

            except Exception:
                logger.exception("Agent graph execution failed for prompt: %r", prompt)
                error_text = "Sorry, something went wrong while processing that. Please try again."
                status_box.update(label="Error", state="error", expanded=True)
                st.error(f"⚠️ {error_text}")
                st.session_state.messages.append({"role": "assistant", "content": error_text})
                st.stop()

        # Extract answer with defensive fallback
        answer_text = (
            result.get("answer")
            or result.get("direct_response")
            or "I wasn't able to generate a response for that — could you try rephrasing?"
        )

        metadata = {
            "raw_question":        prompt,
            "standalone_question": result.get("standalone_question"),
            "route":               result.get("route"),
            "thought":             result.get("thought"),
            "is_sufficient":       result.get("is_sufficient"),
            "retry_count":         result.get("retry_count"),
            "k":                   result.get("k"),
            "sources":             result.get("sources"),
        }

        # Render trace expander and assistant answer
        render_trace(metadata)
        st.markdown(answer_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "metadata": metadata}
        )