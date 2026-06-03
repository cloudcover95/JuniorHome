# path: src/juniorhome/api_server.py
#!/usr/bin/env python3
"""
API Server

Optional lightweight API server for exposing JuniorHome functionality
over HTTP. Useful for integration with other systems or web UIs.
"""

import logging
from typing import Any, Dict, Optional

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


def create_app(orchestrator: Optional[Any] = None) -> Any:
    if not HAS_FASTAPI:
        logging.warning("FastAPI not installed. API server disabled.")
        return None

    app = FastAPI(title="JuniorHome API", version="0.1.0")

    @app.get("/status")
    async def get_status():
        if orchestrator:
            return orchestrator.status()
        return {"status": "ok"}

    @app.post("/llm")
    async def query_llm(payload: Dict[str, Any]):
        if not orchestrator:
            return JSONResponse({"error": "Orchestrator not available"}, status_code=503)

        prompt = payload.get("prompt", "")
        prefer_bitnet = payload.get("prefer_bitnet", False)
        model = payload.get("model", "llama3.2")

        result = orchestrator.route_llm(prompt, prefer_bitnet=prefer_bitnet, model=model)
        return result

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, orchestrator: Optional[Any] = None):
    if not HAS_FASTAPI:
        print("FastAPI is required to run the API server.")
        print("Install with: pip install fastapi uvicorn")
        return

    import uvicorn
    app = create_app(orchestrator)
    logging.info(f"Starting JuniorHome API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
