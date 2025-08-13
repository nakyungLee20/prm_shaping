import json
import time
import os
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# Project-level helpers
from config import PRMConfig
from mi_reward_2 import MIReward2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def jsonl_to_json(jsonl_path, json_path):
    data = read_jsonl(jsonl_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Converted {jsonl_path} to {json_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PRMConfig()
    model_name = "mistralai/Mathstral-7B-v0.1"  
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,          
        bnb_4bit_quant_type="nf4",   
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_compute_dtype=torch.bfloat16  
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",     
        quantization_config=bnb_config,
        torch_dtype="auto",   
        trust_remote_code=True 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    mi = MIReward2(config=cfg, model=model, tokenizer=tokenizer)
    print("Finish loading model and tokenizer")

    output_file = "/home/leena/ccc_eval/mcts_prm/cmi_samples/gsm8k_mi_mistral_full.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, entry in enumerate(mi.gsm8k_reward_dataset_streaming(split="train", start=0, norm = "unit")):
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush() 

    print(f"Data saved to {output_file}")
    jsonl_to_json(output_file, "/home/leena/ccc_eval/mcts_prm/cmi_samples/gsm8k_mi_mistral_full.json")

if __name__ == "__main__":
    main() 