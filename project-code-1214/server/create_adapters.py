import os
import torch
from transformers import LlamaForCausalLM, LlamaConfig
from peft import LoraConfig, get_peft_model

def main():
    print(">>> Creating Adapters for Server...")
    model_id = "NousResearch/Llama-2-7b-hf"
    
    # Config 로드 (모델 다운로드는 vLLM이 실행될 때 자동으로 하거나 여기서 캐싱됨)
    print(f"Loading config: {model_id}")
    config = LlamaConfig.from_pretrained(model_id)
    model = LlamaForCausalLM(config).to(dtype=torch.float16)
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=16, 
        target_modules=["q_proj", "v_proj"], 
        task_type="CAUSAL_LM"
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    # 두 가지 페르소나 어댑터 저장
    base_dir = "./adapters"
    os.makedirs(base_dir, exist_ok=True)
    
    for name in ["art_curator", "chat_bot"]:
        save_path = os.path.join(base_dir, name)
        peft_model.save_pretrained(save_path)
        print(f" - Saved adapter: {save_path}")

if __name__ == "__main__":
    main()
