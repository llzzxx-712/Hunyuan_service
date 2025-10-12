import asyncio
from contextlib import asynccontextmanager

import fastapi
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

queue = asyncio.Queue(maxsize=1)


class TextToImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024


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
    # model = get_model(MODEL_TYPE)
    asyncio.create_task(worker_loop())
    yield
    model.cleanup()


app = fastapi.FastAPI(title="Hunyuan-service", lifespan=lifespan)


@app.post("/infer")
async def infer(req: TextToImageRequest):
    prompt = req.prompt
    return {"message": prompt}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
