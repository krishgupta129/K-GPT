import os
import torch


SOURCE = os.path.join(
    os.path.dirname(__file__),
    "model",
    "weights",
    "gpt2-medium355M-sft.pth",
)

OUTPUT = os.path.join(
    os.path.dirname(__file__),
    "model",
    "weights",
    "gpt2-medium355M-sft-fp16.pth",
)


def main():
    print("=" * 60)
    print("K-GPT vo1 - FP16 Checkpoint Conversion")
    print("=" * 60)

    print("\nLoading original v1 checkpoint...")
    state = torch.load(
        SOURCE,
        map_location="cpu",
        weights_only=True,
    )

    print("Original checkpoint loaded.")

    fp16_state = {}

    for name, tensor in state.items():
        if tensor.is_floating_point():
            fp16_state[name] = tensor.half()
        else:
            fp16_state[name] = tensor

    print("Converted floating-point weights to FP16.")

    torch.save(fp16_state, OUTPUT)

    print("\nvo1 checkpoint saved:")
    print(OUTPUT)

    print("\nOriginal checkpoint size:")
    print(f"{os.path.getsize(SOURCE) / (1024 ** 3):.2f} GB")

    print("vo1 FP16 checkpoint size:")
    print(f"{os.path.getsize(OUTPUT) / (1024 ** 3):.2f} GB")

    print("\nConversion complete.")


if __name__ == "__main__":
    main()
