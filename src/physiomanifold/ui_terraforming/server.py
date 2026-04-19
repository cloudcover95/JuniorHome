# src/physiomanifold/ui_terraforming/server.py

import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.physiomanifold.ui_terraforming.agent import TerraformingAgent

app = FastAPI()
app.mount("/public", StaticFiles(directory="src/physiomanifold/ui_terraforming/public"), name="public")
terraformer = TerraformingAgent()

@app.get("/")
async def serve_canvas():
    return FileResponse("src/physiomanifold/ui_terraforming/public/index.html")

@app.websocket("/ws/audit")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("JuniorHome Audit API Socket Connected.")
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            if payload.get("action") == "terraform":
                await websocket.send_json({"type": "status", "message": "NODE: Compiling component..."})
                
                # Generate new UI board via MLX-LM
                html_component = terraformer.generate_board(
                    system_context="JuniorHome OS. M4 Node, 45W envelope. TDA Mesh active.",
                    component_request=payload["request"]
                )
                
                await websocket.send_json({
                    "type": "board_injection",
                    "html": html_component
                })
                
    except WebSocketDisconnect:
        logging.info("Audit API Socket Disconnected.")
