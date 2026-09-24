from datasets import load_dataset

DATASET_NAME = "juliannunezb/smoltalk-gpt2-sft"


def print_messages(messages):
    print("\nMESSAGES")
    print("-" * 60)

    for i, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = message.get("content", "")

        print(f"\n[{i}] {role.upper()}")
        print(content[:3000])


def inspect_split(dataset, split_name):
    print("\n" + "=" * 80)
    print(f"SPLIT: {split_name}")
    print("=" * 80)

    print("Rows:", len(dataset))
    print("Columns:", dataset.column_names)

    for index in [0, 1, 2, 10, 100]:
        if index >= len(dataset):
            continue

        example = dataset[index]

        print("\n" + "-" * 80)
        print(f"EXAMPLE {index}")
        print("-" * 80)

        print_messages(example["messages"])

        print("\nTEXT")
        print("-" * 60)
        print(example["text"][:3000])

        print("\nTOKEN INFORMATION")
        print("-" * 60)
        print("Token count:", len(example["token_ids"]))
        print("Loss-mask count:", len(example["loss_mask"]))

        mask_sum = sum(example["loss_mask"])

        print("Assistant-loss tokens:", mask_sum)

        if len(example["loss_mask"]) > 0:
            percentage = (mask_sum / len(example["loss_mask"])) * 100
            print(f"Assistant-loss percentage: {percentage:.2f}%")


def main():
    print("=" * 80)
    print("K-GPT v2 SmolTalk Inspection")
    print("=" * 80)

    print("\nLoading cached dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded successfully.")
    print(dataset)

    for split_name, split in dataset.items():
        inspect_split(split, split_name)


if __name__ == "__main__":
    main()
