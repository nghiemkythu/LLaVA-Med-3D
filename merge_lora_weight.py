from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel 
import torch

path = "/netscratch/duynguyen/Research/medllm_new/llama2/weights/original_llama2_weights/models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8"
peft_model_id = "/netscratch/duynguyen/Research/medllm_new/llama2/results_modified/checkpoint-341292"

base_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

model = PeftModel.from_pretrained(base_model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id, use_fast=True)

merged_model = model.merge_and_unload() 

merged_model.save_pretrained("./weights_full/llama_med")
tokenizer.save_pretrained("./weights_full/llama_med")