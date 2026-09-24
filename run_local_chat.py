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
from model.prompt_template import build_chat_prompt


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
    "k_gpt_v2_beta_deploy.pth",
)


def generate_response(model, instruction, history, device, max_new_tokens=120):
    prompt = build_chat_prompt(instruction, history)

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

    return clean_response(response)


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

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

    print("K-GPT v2 Beta loaded successfully.\n")

    return model, device


def main():
    model, device = load_model()

    history = []

    print("=" * 60)
    print("                 K-GPT v2 Beta")
    print("                    Chat Mode")
    print("=" * 60)
    print("Universal K-GPT Chat Template enabled.")
    print("Commands: /exit, /quit, /clear")
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

        if user_input.lower() == "/clear":
            history = []
            print("\nConversation cleared.")
            continue

        try:
            response = generate_response(
                model,
                user_input,
                history,
                device,
                max_new_tokens=120,
            )

            print("\nK-GPT:", response)

            history.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
                print("\nGPU memory ran out. Try a shorter conversation.")
            else:
                raise


if __name__ == "__main__":
    main()
