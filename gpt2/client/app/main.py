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
import argparse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends, HTTPException, status



cache_dir = "./model_store/"
security = HTTPBasic()
correct_username = "UStAilaN"
correct_password = "pK9#mJ4$xL2@"

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ModelService:
    def __init__(self, args):
        # Use the full model name from HuggingFace
        self.model_executor=ModelExecutor(args)

    async def generate(self, batch: List[GenerationRequest],):
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

def create_app(args,model_service):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.model_service = model_service
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
    async def generate(request: GenerationRequest, username: str = Depends(authenticate)):
        try:
            result = await app.batch_manager.add_request(request)
            return {"generated_text": await result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='gpt2', )

    parser.add_argument('--rank', type=int, default=0, )
    parser.add_argument('--local_rank', type=int, default=0, )
    parser.add_argument('--world_size', type=int, default=2, )

    parser.add_argument('--master_address', type=str, default="192.168.1.153", )
    parser.add_argument('--master_port', type=str, default="29500", )
    parser.add_argument('--device', type=str, default="cuda", )
    parser.add_argument('--ifname', type=str, default="ens5", )

    
    args = parser.parse_args()
    
    model_service = ModelService(args)

    fastapi_app = create_app(args,model_service)

    uvicorn.run(
        #"main:app",
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="uvloop"
    )