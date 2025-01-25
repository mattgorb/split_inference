from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from contextlib import asynccontextmanager
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from model_executor import ModelExecutor


"""
export NCCL_SOCKET_IFNAME=en0
export MASTER_ADDR="192.168.1.153"
export MASTER_PORT="29500"
export WORLD_SIZE="2"
export LOCAL_RANK="0"
export RANK="0"

lsof -ti :29500 | xargs -r kill -9

module purge
module load python/anaconda



"""

cache_dir = "./model_store/"

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ModelService:
    def __init__(self):
        # Use the full model name from HuggingFace
        model_name = "gpt2"  # or "openai-community/gpt2"
        self.model_executor=ModelExecutor(model_name='gpt2')

    async def generate(self, batch: List[GenerationRequest]):
        # Process batch of requests
        prompts = [req.prompt for req in batch]
        max_lengths = [req.max_length for req in batch]
        temperatures = [req.temperature for req in batch]

        responses=self.model_executor.run(prompts, max_lengths, temperatures)

        return responses
class BatchManager:
    def __init__(self, max_batch_size=32, max_wait_time=0.1):
        self.queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

    async def add_request(self, request: GenerationRequest):
        future = asyncio.Future()
        await self.queue.put((request, future))
        return future

    async def process_batches(self, inference_fn):
        while True:
            batch = []
            futures = []

            # Get first request
            request, future = await self.queue.get()
            batch.append(request)
            futures.append(future)

            # Accumulate batch
            timeout = asyncio.create_task(asyncio.sleep(self.max_wait_time))
            
            while len(batch) < self.max_batch_size and not timeout.done():
                try:
                    request, future = await asyncio.wait_for(
                        self.queue.get(), 
                        timeout=self.max_wait_time
                    )
                    batch.append(request)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break

            # Process batch
            try:
                results = await inference_fn(batch)
                for future, result in zip(futures, results):
                    future.set_result(result)
            except Exception as e:
                for future in futures:
                    future.set_exception(e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model and batch manager
    app.model_service = ModelService()
    app.batch_manager = BatchManager()
    app.inference_task = asyncio.create_task(
        app.batch_manager.process_batches(app.model_service.generate)
    )
    yield
    # Cleanup
    app.inference_task.cancel()
    try:
        await app.inference_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        result = await app.batch_manager.add_request(request)
        return {"generated_text": await result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="uvloop"
    )