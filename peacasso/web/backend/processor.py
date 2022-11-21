
import copy
import json
from typing import List
from dataclasses import asdict
from fastapi import WebSocket
from peacasso.datamodel import GeneratorConfig, SocketData
from peacasso.generator import ImageGenerator
from peacasso.utils import base64_to_pil, pil_to_base64, sanitize_config
import traceback


# enable web socket connection for real time updates
class ConnectionManager:
    """Connection manager for websockets"""

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


async def process_request(request: SocketData, generator: ImageGenerator, websocket: WebSocket):
    """Process socket request"""

    if request.type == "generate":
        prompt_config = GeneratorConfig(**request.data["config"])
        sanitized_config = asdict(sanitize_config(copy.deepcopy(prompt_config)))

        # def generator_callback(i: int, t: int, latents: torch.FloatTensor, images: PIL.Image.Image):
        #     """Callback for generator"""

        #     asyncio.get_event_loop().create_task(websocket.send_text(json.dumps(
        #         {"type": "generator_progress", "data": {"i": i, "config": sanitized_config}})))

        # generator.generate(request.data, socket=socket)
        if prompt_config.init_image is not None:
            prompt_config.init_image, _ = base64_to_pil(prompt_config.init_image)
        if prompt_config.mask_image:
            _, prompt_config.mask_image = base64_to_pil(prompt_config.mask_image)
        response = None
        # prompt_config.callback = generator_callback
        try:
            # print("prompt_config >>>>>> ", prompt_config)
            result = generator.generate(prompt_config)
            images = []
            for image in result["images"]:
                # convert pil image to base64 and prepend with data uri
                images.append("data:image/png;base64, " + pil_to_base64(image))
            result["images"] = images
            # print(result)
            response = json.dumps({"type": "generate_complete", "data": {
                "status": {"status": True, "message": "success"},
                "config": sanitized_config, "result": result}})  # send result to client
        except Exception as e:
            # print("error: {}".format(e))
            traceback.print_exc()
            response = json.dumps({"type": "generate_complete", "data": {
                "status": {"status": False, "message": str(e)},
                "config": sanitized_config, "result": None}})
        await websocket.send_text(response)

    elif request.type == "ping":
        response = json.dumps({"type": "ping", "data": "pong"})
        await websocket.send_text(response)
        # generator.cancel()
