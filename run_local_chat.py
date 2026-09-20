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


def format_prompt(user_input):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{user_input}"
        "\n\n### Response:\n"
    )


def generate_response(model, user_input, device, max_new_tokens=120):
    prompt = format_prompt(user_input)

    token_ids = text_to_token_ids(prompt, tokenizer).to(device)

    with torch.inference_mode():
        output_ids = generate(
            model=model,
            idx=token_ids,
            max_new_tokens=max_new_tokens,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )

    generated_text = token_ids_to_text(output_ids, tokenizer)

    response = generated_text[len(prompt):]
    response = response.replace("### Response:", "").strip()

    return response


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading K-GPT checkpoint...")

    model = GPTModel(BASE_CONFIG)

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print("Model loaded successfully.\n")

    return model, device


def main():
    model, device = load_model()

    print("=" * 60)
    print("                     K-GPT")
    print("              GPT-2 Medium 355M")
    print("=" * 60)
    print("Type your instruction and press Enter.")
    print("Commands: /exit, /quit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nK-GPT: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "/quit"}:
            print("\nK-GPT: Goodbye!")
            break

        print("\nK-GPT: ", end="", flush=True)

        try:
            response = generate_response(
                model,
                user_input,
                device,
                max_new_tokens=120,
            )
            print(response)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
                print("\nGPU memory ran out. Try a shorter prompt.")
            else:
                raise


if __name__ == "__main__":
    main()
