import asyncio
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.model_registry import get_model
from src.models.hunyuan import TextToImageInput
from src.models.qwen2_5 import ImageToTextInput
from src.utils import image_to_base64

MODEL_TYPE = os.getenv("MODEL_TYPE", "text_to_image").lower()

queue = asyncio.Queue(maxsize=1)

model = None


class TextToImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 768
    steps: int = 50


class ImageToTextRequest(BaseModel):
    imgs: list[str]
    prompt: str


async def worker_loop():
    while True:
        func, args, kwargs, fut = await queue.get()
        try:
            result = await asyncio.to_thread(func, *args, **kwargs)
            fut.set_result(result)
        except Exception as e:
            fut.set_exception(e)
        finally:
            queue.task_done()


async def enqueue(func, *args, **kwargs):
    fut = asyncio.Future()
    await queue.put((func, args, kwargs, fut))
    return await fut


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = get_model(MODEL_TYPE)
    asyncio.create_task(worker_loop())
    yield


app = FastAPI(title="Hunyuan-service", lifespan=lifespan)


def text_to_image_service():
    @app.post("/infer")
    async def infer(req: TextToImageRequest):
        prompt = req.prompt
        width = req.width
        height = req.height
        steps = req.steps
        input = TextToImageInput(prompt=prompt)
        output = await enqueue(
            model.infer, input, width=width, height=height, num_inference_steps=steps
        )
        return {"image_base64": image_to_base64(output.image)}


def image_to_text_service():
    @app.post("/infer")
    async def infer(req: ImageToTextRequest):
        images = req.imgs
        prompt = req.prompt
        input = ImageToTextInput(imgs=images, prompt=prompt)
        output = await enqueue(model.infer, input)
        return {"text": output.result}


if MODEL_TYPE == "text_to_image":
    text_to_image_service()
elif MODEL_TYPE == "image_to_text":
    image_to_text_service()
else:
    raise ValueError(f"Unsupported model type: {MODEL_TYPE}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
