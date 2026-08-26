"""FastAPI Application Server Entrypoint for Kara Agentic RAG Assistant.

Run directly via:
    uv run app.py
    # or
    python app.py
"""

import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" ☁️  Kara — Dual-Source Agentic RAG Assistant API")
    print("=" * 60)
    print(" 🚀 Server running at: http://127.0.0.1:8000")
    print(" 📖 Interactive Docs: http://127.0.0.1:8000/api/v1/docs")
    print(" 🔍 ReDoc:            http://127.0.0.1:8000/api/v1/redoc")
    print("=" * 60 + "\n")

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )