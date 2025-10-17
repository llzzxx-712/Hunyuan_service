import asyncio
import os
from contextlib import asynccontextmanager
from time import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.model_registry import get_model
from src.models.hunyuan import TextToImageInput
from src.models.qwen2_5 import ImageToTextInput
from src.utils import image_to_base64

MODEL_TYPE = os.getenv("MODEL_TYPE", "text_to_image").lower()
IMAGE2TEXT_BATCH_SIZE = int(os.getenv("IMAGE2TEXT_BATCH_SIZE", 5))
TEXT2IMAGE_BATCH_SIZE = int(os.getenv("TEXT2IMAGE_BATCH_SIZE", 5))
BATCH_TIMEOUT = 0.3

queue = asyncio.Queue(maxsize=64)

model = None


class TextToImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 768
    steps: int = 50


class ImageToTextRequest(BaseModel):
    imgs: list[str]
    prompt: str


async def batch_worker():
    batch_size = TEXT2IMAGE_BATCH_SIZE if MODEL_TYPE == "text_to_image" else IMAGE2TEXT_BATCH_SIZE
    print(f"[Worker] Started with batch size {batch_size} and timeout {BATCH_TIMEOUT}")

    while True:
        first_task = await queue.get()
        batch = [first_task]
        start_time = time()

        while len(batch) < batch_size and (time() - start_time) < BATCH_TIMEOUT:
            print(f"[Worker] Queue size: {queue.qsize()}")
            try:
                task = queue.get_nowait()
                batch.append(task)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)

        funcs, inputs, futs = zip(*batch)
        print(f"[Worker] Processed {len(batch)} tasks")

        try:
            results = await asyncio.to_thread(funcs[0], inputs)
            print(f"[Worker] Processed {len(results)} results")
            if not isinstance(results, list):
                results = [results]
            for fut, result in zip(futs, results):
                fut.set_result(result)
        except Exception as e:
            for fut in futs:
                fut.set_exception(e)
                print(f"[Worker] Error: {e}")
        finally:
            for task in batch:
                queue.task_done()


async def enqueue(func, input_data):
    fut = asyncio.get_event_loop().create_future()
    await queue.put((func, input_data, fut))
    return await fut


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = get_model(MODEL_TYPE)
    asyncio.create_task(batch_worker())
    yield


app = FastAPI(title="Hunyuan-service-batch", lifespan=lifespan)


def text_to_image_batch_service():
    @app.post("/infer")
    async def infer(req: TextToImageRequest):
        prompt = req.prompt
        # width = req.width
        # height = req.height
        # steps = req.steps
        input = TextToImageInput(prompt=prompt)
        output = await enqueue(model.batch_infer, input)
        return {"image_base64": image_to_base64(output.image)}


def image_to_text_batch_service():
    @app.post("/infer")
    async def infer(req: ImageToTextRequest):
        images = req.imgs
        prompt = req.prompt
        input = ImageToTextInput(imgs=images, prompt=prompt)
        output = await enqueue(model.batch_infer, input)
        return {"text": output.result}


if MODEL_TYPE == "text_to_image":
    text_to_image_batch_service()
elif MODEL_TYPE == "image_to_text":
    image_to_text_batch_service()
else:
    raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
