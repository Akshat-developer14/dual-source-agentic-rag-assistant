# ☁️ Kara — Dual-Source Self-Reflective Agentic RAG Assistant

[![Python](https://img.shields.io/badge/Python-3.11%2B%20%7C%203.14-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2015-000000.svg?logo=next.js&logoColor=white)](https://nextjs.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/LLM%20Inference-Groq%20LPU-F55036.svg)](https://groq.com/)
[![PostgreSQL](https://img.shields.io/badge/Database-Supabase%20Postgres-3ECF8E.svg?logo=supabase&logoColor=white)](https://supabase.com/)
[![Vector Store](https://img.shields.io/badge/VectorStore-FAISS-00599C.svg)](https://github.com/facebookresearch/faiss)
[![Search Engine](https://img.shields.io/badge/Live%20Search-Tavily%20AI-4F46E5.svg)](https://tavily.com/)
[![Evaluation](https://img.shields.io/badge/Evals-LLM--as--a--Judge-8A2BE2.svg)](evals/EVAL_REPORT.md)

An enterprise-grade, multi-tenant **Agentic Retrieval-Augmented Generation (RAG)** platform specializing in **AWS Cloud Architecture & Infrastructure Intelligence**. 

**Kara** combines static enterprise documentation from the **AWS Well-Architected Framework** with dynamic, real-time web retrieval via **Tavily AI**. Built on a self-corrective **LangGraph StateGraph**, the system autonomously evaluates retrieved context, dynamically tunes retrieval density ($k$), reformulates queries based on grader feedback, and falls back to live web intelligence when internal documentation is incomplete.

---

## 📊 Benchmark Evaluation Scorecard

Evaluated against our curated golden benchmark dataset using `openai/gpt-oss-120b` as an automated **LLM-as-a-Judge**:

| Metric | Benchmark Score | Target Threshold | Status |
| :--- | :--- | :--- | :--- |
| **🧭 Routing Accuracy** | **100.0%** | $\ge 90\%$ | ✅ **PASS** |
| **🛡️ Faithfulness (Anti-Hallucination)** | **98.6%** | $\ge 90\%$ | ✅ **PASS** |
| **🎯 Answer Relevance** | **100.0%** | $\ge 85\%$ | ✅ **PASS** |
| **📚 Citation Compliance** | **100.0%** | $\ge 90\%$ | ✅ **PASS** |
| **⚡ Token Efficiency** | **~5,180 tokens/turn** | $< 6,000\text{ tokens}$ | ✅ **OPTIMAL** |

*See the full report at [evals/EVAL_REPORT.md](evals/EVAL_REPORT.md).*

---

## 🚀 Key Highlights & Architectural Features

- **🧭 Intelligent Multi-Route Classifier**: Evaluates standalone queries with Chain-of-Thought (CoT) reasoning to route into `internal` (AWS docs), `web` (pricing/outages/hardware comparisons), `chitchat` (conversational), or `unrelated` (domain guardrail deflection).
- **🔁 Self-Corrective RAG (CRAG) Feedback Loop**: Evaluates context completeness via a dedicated Grader LLM. If the retrieved context is insufficient, the system reformulates the query using targeted grader notes and retries retrieval before falling back to web search.
- **📊 Dynamic Retrieval Density ($k$-tuning)**: The Grader LLM dynamically scales chunk density ($k \in [4, 10]$) based on whether context was merely truncated vs. missing.
- **🔒 Enterprise Security & Multi-Tenancy**: Stateless JWT authentication (`HS256`), native Bcrypt password hashing, and user-isolated conversation threads stored in **Supabase PostgreSQL**.
- **⚡ Real-Time Server-Sent Events (SSE)**: Streams step-by-step node execution updates, active reasoning traces, and cumulative token metrics in real time.
- **📈 End-to-End Token Tracking**: Tracks prompt, completion, and total token usage across all graph hops, persisted in PostgreSQL and rendered live in the UI.
- **💻 Modern SaaS Frontend**: Decoupled **Next.js 15 (Turbopack)** web client featuring a landing page, JWT auth modal, multi-threaded conversation sidebar, markdown rendering, and collapsible Agent Trace drawer.

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    Client["💻 Next.js 15 Frontend"] <-->|Server-Sent Events SSE| API["⚡ FastAPI Backend Service"]
    API <-->|SQLAlchemy NullPool| DB[("🐘 Supabase PostgreSQL")]
    
    subgraph LangGraph_Agent ["LangGraph Cyclical State Machine"]
        Start([🚀 START]) --> Ctx[🔤 contextualize_node\nResolve Chat History & Pronouns]
        Ctx --> Router[🧭 router_node\nClassify Intent with CoT Reasoning]
        
        Router -->|chitchat / unrelated| DirectResp([🛑 Direct Response / Deflection])
        Router -->|web| WebSearch[🌐 web_search_node\nTavily API Search]
        Router -->|internal| Retriever[📂 retriever_node\nFAISS Dense Retrieval Dynamic k]
        
        Retriever --> Grader[🔬 grade_context_node\nCheck Completeness & Recommend k]
        
        Grader -->|Context Sufficient ✅| Synth[🧠 synthesizer_node\nGround Answer & Attribute Sources]
        Grader -->|Insufficient & Retry < 1 ❌| Rewriter[✏️ rewrite_query_node\nRefine Query with Grader Notes]
        Grader -->|Insufficient & Retries Exhausted ⚠️| WebSearch
        
        Rewriter -->|Retry Search| Retriever
        WebSearch --> Synth
        Synth --> End([🏁 Final Answer + Sources + Token Metrics])
    end

    API --> LangGraph_Agent
```

---

## 🔬 Graph Node Execution Breakdown

| Node Name | Primary Model | Fallback Model | Responsibility |
| :--- | :--- | :--- | :--- |
| **`contextualize_node`** | `openai/gpt-oss-20b` | `openai/gpt-oss-120b` | Resolves pronouns and references across previous turns into standalone queries. |
| **`router_node`** | `openai/gpt-oss-20b` | `openai/gpt-oss-120b` | Classifies query into `internal`, `web`, `chitchat`, or `unrelated` with JSON schema enforcement. |
| **`retriever_node`** | *Local Embedding* | — | Retrieves top-$k$ document chunks from local FAISS index (`all-MiniLM-L6-v2`) with MD5 deduplication. |
| **`grade_context_node`** | `openai/gpt-oss-120b` | `openai/gpt-oss-20b` | Evaluates context sufficiency under zero-outside-knowledge assumptions and adjusts target $k$. |
| **`rewrite_query_node`** | `openai/gpt-oss-20b` | `openai/gpt-oss-120b` | Reformulates search query using diagnostic grader feedback; increments `retry_count`. |
| **`web_search_node`** | *Tavily API* | — | Fetches real-time web context with exponential backoff; updates route to `web` on fallback. |
| **`synthesizer_node`** | `openai/gpt-oss-120b` | `openai/gpt-oss-20b` | Synthesizes grounded answer strictly from cited context with JSON schema validation. |

---

## 📂 Project Structure

```text
dual-source-agentic-rag-assistant/
├── backend/
│   └── app/
│       ├── api/v1/
│       │   ├── endpoints/
│       │   │   ├── auth.py          # /register, /login, /me endpoints
│       │   │   └── chat.py          # /conversations, /stream SSE endpoints
│       │   └── api.py               # APIRouter registration
│       ├── core/
│       │   ├── config.py            # Pydantic Settings & env configuration
│       │   ├── deps.py              # FastAPI auth & DB dependency injection
│       │   └── security.py          # Bcrypt hashing & PyJWT token management
│       ├── db/
│       │   ├── base.py              # SQLAlchemy 2.0 DeclarativeBase
│       │   └── session.py           # Engine with NullPool & session generator
│       ├── models/
│       │   ├── chat.py              # Conversation & Message ORM models
│       │   └── user.py              # User ORM model
│       ├── schemas/
│       │   ├── chat.py              # Pydantic request/response/stream schemas
│       │   └── user.py              # User & JWT Token schemas
│       ├── services/
│       │   ├── chat_service.py      # Conversation CRUD & LangGraph SSE bridge
│       │   └── user_service.py      # User authentication & registration CRUD
│       └── main.py                  # FastAPI app entrypoint with CORS
├── data/
│   └── aws_well_architected.pdf     # Primary internal knowledge base document
├── evals/
│   ├── dataset.json                 # Golden benchmark evaluation dataset
│   ├── evaluate_rag.py              # Automated LLM-as-a-Judge evaluation runner
│   ├── eval_results.json            # Structured benchmark test results
│   └── EVAL_REPORT.md               # Formatted markdown evaluation report
├── src/
│   ├── graph.py                     # LangGraph StateGraph & conditional edges
│   ├── nodes.py                     # Agent nodes with token extraction & error fallbacks
│   ├── prompts.py                   # System prompts & few-shot JSON templates
│   ├── state.py                     # TypedDict AgentState schema
│   └── tools.py                     # FAISS retriever cache & Tavily search client
├── vectorstore/
│   └── faiss_index/                 # Persisted local FAISS index & metadata
├── app.py                           # FastAPI Uvicorn launcher (uv run app.py)
├── ingest.py                        # Document chunking & FAISS builder
├── pyproject.toml                   # Project dependencies and packaging
├── uv.lock                          # Deterministic dependency lockfile
└── README.md                        # Project documentation
```

---

## 🛠️ Setup & Installation Guide

### Prerequisites
- Python **3.11+** or **3.14**
- [**uv**](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- A [Groq API Key](https://console.groq.com/)
- A [Tavily API Key](https://app.tavily.com/)
- A [Supabase PostgreSQL Database](https://supabase.com/)

---

### 1. Clone the Repository

```bash
git clone https://github.com/Akshat-developer14/dual-source-agentic-rag-assistant.git
cd dual-source-agentic-rag-assistant
```

---

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
TAVILY_API_KEY=tvly-your_tavily_api_key_here
DATABASE_URL=postgresql+psycopg://postgres.xxxx:your_password@aws-0-region.pooler.supabase.com:6543/postgres?sslmode=require
SECRET_KEY=your-secure-jwt-secret-key
```

---

### 3. Install Dependencies

Using `uv` (Fast & Deterministic):

```bash
uv sync
```

---

### 4. Build the FAISS Vector Store

Parse the AWS Well-Architected PDF, generate embeddings, and build the local FAISS index:

```bash
uv run ingest.py
```

---

### 5. Launch the FastAPI Backend

```bash
uv run app.py
```

- **Backend API**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/api/v1/docs`
- **Health Check**: `http://localhost:8000/health`

---

### 6. Run the Benchmark Evaluation Suite

Execute the automated evaluation suite against the golden dataset:

```bash
uv run python -m evals.evaluate_rag
```

This will run all test cases, evaluate via `openai/gpt-oss-120b` (LLM-as-a-Judge), and generate an updated [evals/EVAL_REPORT.md](evals/EVAL_REPORT.md).

---

## 🌐 Next.js Frontend Client

The frontend is housed in a separate repository:
👉 [agentic-rag-assistant-frontend](https://github.com/Akshat-developer14/agentic-rag-assistant-frontend)

To run the frontend:
```bash
cd ../agentic-rag-assistant-frontend
yarn install
yarn dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 👤 Author

**Akshat**
- GitHub: [@Akshat-developer14](https://github.com/Akshat-developer14)

---

## 📄 License

This project is licensed under the **MIT License**. Internal reference documentation includes the [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/) whitepaper.
