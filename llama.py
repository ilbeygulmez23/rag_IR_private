# llama.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

print("Loading Turkish LLAMA 8B...")

model_path = "ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1"  # Hugging Face model hub

# Load once
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Loading completed.")


def respond(prompt, task="generate"):
    with torch.no_grad():
        max_tokens = 64 if task == "select" else 512
        output = pipe(
            prompt,
            max_new_tokens=max_tokens,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.empty_cache()
    return output[0]["generated_text"].strip()
