import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.lora.request import LoRARequest

# 1. 모델 및 어댑터 설정
MODEL_NAME = "NousResearch/Llama-2-7b-hf"
ADAPTER_DIR = "./adapters" 

app = FastAPI()

# 2. vLLM 엔진 초기화
print(">>> Initializing vLLM Engine with Chunked Prefill & Multi-LoRA...")

engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    dtype="auto",
    gpu_memory_utilization=0.9,
    enable_chunked_prefill=True,
    max_num_batched_tokens=512,
    enable_lora=True,
    max_loras=4,
    max_lora_rank=16,
    disable_log_stats=True
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# 3. 데이터 구조 정의
class ChatRequest(BaseModel):
    prompt: str
    adapter_type: str = "chat_bot"
    max_tokens: int = 128

# 4. 추론 엔드포인트
@app.post("/generate")
async def generate_response(request: ChatRequest):
    try:
        adapter_path = os.path.join(ADAPTER_DIR, request.adapter_type)
        if not os.path.exists(adapter_path):
            raise HTTPException(status_code=400, detail=f"Adapter '{request.adapter_type}' not found.")

        lora_id = 1 if request.adapter_type == "art_curator" else 2
        lora_req = LoRARequest(request.adapter_type, lora_id, adapter_path)
        
        sampling_params = SamplingParams(temperature=0.7, max_tokens=request.max_tokens)
        request_id = f"req-{os.urandom(4).hex()}"
        
        results_generator = engine.generate(
            request.prompt, 
            sampling_params, 
            request_id=request_id, 
            lora_request=lora_req
        )

        final_output = ""
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text

        return {"status": "success", "response": final_output}

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return {"status": "error", "detail": str(e)}

if __name__ == "__main__":
    # [수정 완료] 포트를 8080으로 통일했습니다.
    print(">>> Starting API Server on Port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
