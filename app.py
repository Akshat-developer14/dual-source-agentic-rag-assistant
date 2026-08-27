"""Entrypoint launcher for the Kara Agentic RAG FastAPI Backend."""

import os
import sys
import uvicorn

# Ensure UTF-8 console output for cross-platform and Windows terminal compatibility
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("\n" + "=" * 65)
    print("  KARA AGENTIC RAG ASSISTANT — FASTAPI BACKEND SERVICE")
    print("=" * 65)
    print(f"  Server URL      : http://localhost:{port}")
    print(f"  API Docs (OpenAPI): http://localhost:{port}/api/v1/docs")
    print(f"  Health Endpoint : http://localhost:{port}/health")
    print("=" * 65 + "\n")

    uvicorn.run(
        "backend.app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )