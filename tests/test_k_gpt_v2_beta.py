import os
import torch

from model.model import GPTModel
from model.generate import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
    clean_response,
)
from model.tokenizer import tokenizer
from model.prompt_template import build_prompt


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
    os.path.dirname(os.path.dirname(__file__)),
    "model",
    "weights",
    "k_gpt_v2_beta_deploy.pth",
)

TESTS = [
    ("AI Definition", "What is artificial intelligence?"),
    ("Algorithm", "What is an algorithm?"),
    ("Arithmetic", "What is 1 + 1?"),
    ("Python", "What is Python used for?"),
    ("Machine Learning", "Explain machine learning in simple terms."),
    ("GPU", "What is a GPU?"),
    ("Programming", "Write a Python function that adds two numbers."),
    ("Reasoning", "If I have 3 apples and get 2 more, how many apples do I have?"),
    ("Geography", "What is the capital of France?"),
    ("Science", "What planet do humans live on?"),
    ("Instruction", "Give me three uses of artificial intelligence."),
    ("Short Answer", "Complete this sentence: The sky is"),
    ("Explanation", "Why is data important in machine learning?"),
    ("Definition", "What is overfitting in machine learning?"),
    ("Identity", "My name is Krish. What is your name?"),
]


def load_model(device):
    print("Loading K-GPT v2 Beta...")

    model = GPTModel(BASE_CONFIG)

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print("K-GPT v2 Beta loaded successfully.")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    return model


def generate_response(model, instruction, device):
    prompt = build_prompt(instruction)

    token_ids = text_to_token_ids(
        prompt,
        tokenizer,
    ).to(device)

    with torch.inference_mode():
        output_ids = generate(
            model=model,
            idx=token_ids,
            max_new_tokens=80,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )

    generated_text = token_ids_to_text(
        output_ids,
        tokenizer,
    )

    response = generated_text[len(prompt):]

    return clean_response(response)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    print("\n")
    print("=" * 80)
    print("K-GPT v2 BETA EVALUATION")
    print("=" * 80)

    for number, (category, question) in enumerate(TESTS, 1):
        print(f"\n[{number:02d}] {category}")
        print(f"Q: {question}")

        try:
            response = generate_response(
                model,
                question,
                device,
            )

            print(f"A: {response}")

        except Exception as exc:
            print(f"ERROR: {exc}")

    print("\n")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
