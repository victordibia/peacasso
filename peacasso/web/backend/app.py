from io import BytesIO
import io
import zipfile
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os
from peacasso.generator import ImageGenerator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from peacasso.datamodel import GeneratorConfig
import hashlib

from peacasso.utils import base64_to_pil

# # load token from .env variable
hf_token = os.environ.get("HF_API_TOKEN")
generator = ImageGenerator(token=hf_token)

app = FastAPI()
# allow cross origin requests for testing on localhost:800* ports only
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

# if not os.path.exists(static_folder_root):
#     assert False, "Static folder not found: {}. Ensure the front end is built".format(
#         static_folder_root
#     )


# mount peacasso front end UI files
app.mount("/", StaticFiles(directory=static_folder_root, html=True), name="ui")
api.mount("/files", StaticFiles(directory=files_static_root, html=True), name="files")


@api.post("/generate")
def generate(prompt_config: GeneratorConfig) -> str:
    """Generate an image given some prompt"""
    # print(prompt_config.init_image)
    if prompt_config.init_image:
        prompt_config.init_image = base64_to_pil(prompt_config.init_image)
    result = None
    try:
        result = generator.generate(prompt_config)
    except Exception as e:
        return {"status": False, "status_message": str(e)}
    try:
        slug = hashlib.sha256(str(prompt_config).encode("utf-8")).hexdigest()
        zip_io = BytesIO()
        with zipfile.ZipFile(
            zip_io, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as temp_zip:
            for i, image in enumerate(result["images"]):
                zip_path = os.path.join("/", str(slug) + "_" + str(i) + ".png")
                # Add file, at correct path
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                image.close()
                temp_zip.writestr(zip_path, img_byte_arr.getvalue())
        return StreamingResponse(
            iter([zip_io.getvalue()]),
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": f"attachment; filename=images.zip"},
        )
    except Exception as e:
        print("error: {}".format(e))
        return {"status": False, "status_message": str(e)}


@api.get("/cuda")
def list_cuda():
    return generator.list_cuda()
