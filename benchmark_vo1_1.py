import os
import time
import psutil
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

PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explain machine learning in simple terms.

### Response:
"""


def memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def main():
    print("=" * 60)
    print("K-GPT vo1.1 MEMORY-EFFICIENT FP32 LOAD")
    print("=" * 60)

    print(f"Initial RAM: {memory_mb():.2f} MB")

    start = time.perf_counter()

    with torch.device("meta"):
        model = GPTModel(BASE_CONFIG)

    print(f"After meta model creation: {memory_mb():.2f} MB")

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    print(f"After checkpoint load: {memory_mb():.2f} MB")

    model.load_state_dict(state, assign=True)

    print(f"After assign=True: {memory_mb():.2f} MB")

    del state

    print(f"After deleting state_dict: {memory_mb():.2f} MB")

    model.eval()

    load_time = time.perf_counter() - start

    print(f"Model load time: {load_time:.2f} seconds")

    token_ids = text_to_token_ids(PROMPT, tokenizer)

    start = time.perf_counter()

    with torch.inference_mode():
        output_ids = generate(
            model=model,
            idx=token_ids,
            max_new_tokens=50,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )

    generation_time = time.perf_counter() - start

    generated = token_ids_to_text(output_ids, tokenizer)
    response = generated[len(PROMPT):].replace(
        "### Response:",
        "",
        1,
    ).strip()

    print(f"After generation: {memory_mb():.2f} MB")
    print(f"Generation time: {generation_time:.2f} seconds")

    print("\nK-GPT vo1.1 response:")
    print(response)

    print("\n" + "=" * 60)
    print("vo1.1 BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
