import os
import torch

from model.model import GPTModel
from model.generate import generate, text_to_token_ids, token_ids_to_text
from model.tokenizer import tokenizer

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16,
}

CHECKPOINT = os.path.join(
    os.path.dirname(__file__),
    "model",
    "weights",
    "gpt2-medium355M-sft.pth",
)

def extract_response(generated_text, input_text):
    return generated_text[len(input_text):].replace("### Response:", "").strip()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading K-GPT checkpoint...")
    model = GPTModel(BASE_CONFIG)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explain machine learning in simple terms.
"""

    token_ids = text_to_token_ids(prompt, tokenizer).to(device)

    with torch.inference_mode():
        output_ids = generate(
            model=model,
            idx=token_ids,
            max_new_tokens=50,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )

    generated = token_ids_to_text(output_ids, tokenizer)
    response = extract_response(generated, prompt)
    print("\nK-GPT response:\n")
    print(response)

if __name__ == "__main__":
    main()
