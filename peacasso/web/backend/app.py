import json
import logging
from typing import List
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import os
from peacasso.generator import ImageGenerator
from fastapi.middleware.cors import CORSMiddleware
from peacasso.datamodel import ModelConfig, SocketData
from peacasso.web.backend.processor import process_request
import uvicorn
import os

logger = logging.getLogger("peacasso")

assert os.environ.get("HF_API_TOKEN") is not None, "HF_API_TOKEN not set"

# load model using env variables
model_config = ModelConfig(
    model=os.environ.get("PEACASSO_MODEL", "runwayml/stable-diffusion-v1-5"),
    revision=os.environ.get("PEACASSO_REVISION", "fp16"),
    device=os.environ.get("PEACASSO_DEVICE", "cuda:0"),
    token=os.environ.get("HF_API_TOKEN")
)
logger.info(
    ">> Loading Imagenerator pipeline with config. " + str(model_config))
generator = ImageGenerator(model_config)
logger.info(">> Imagenerator pipeline loaded.")

app = FastAPI()
# allow cross origin requests for testing on localhost: 800 * ports only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api = FastAPI(root_path="/api")
app.mount("/api", api)


root_file_path = os.path.dirname(os.path.abspath(__file__))
static_folder_root = os.path.join(root_file_path, "ui")
files_static_root = os.path.join(root_file_path, "files/")

os.makedirs(files_static_root, exist_ok=True)

# mount peacasso front end UI files
app.mount("/", StaticFiles(directory=static_folder_root, html=True), name="ui")
api.mount("/files", StaticFiles(directory=files_static_root, html=True), name="files")


# enable web socket connection for real time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@api.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # parse data to json
            try:
                request = SocketData(**json.loads(data))
                await process_request(request, generator, websocket)
                # await manager.send_personal_message(response, websocket)
            except Exception as e:
                print("error: {}".format(e))
                response = json.dumps({"type": "generate_complete", "data": {
                    "status": {"status": False, "message": str("{}".format(e))},
                }})
                await websocket.send_text(response)
                continue

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client  left the chat")
