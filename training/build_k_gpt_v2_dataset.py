import os
import json
import random
import shutil
from collections import Counter

from datasets import load_dataset, Dataset


SOURCE_DATASET = "juliannunezb/smoltalk-gpt2-sft"

OUTPUT_DIR = "training/k_gpt_v2_dataset"

MAX_CONTEXT_LENGTH = 1024
MIN_ASSISTANT_TOKENS = 16

RANDOM_SEED = 42

TARGET_TOTAL = 400_000
VALIDATION_SIZE = 5_000

LONG_SEQUENCE_STRIDE = 768

TRAIN_SHARD_SIZE = 10_000
VAL_SHARD_SIZE = 1_000


TARGETS = {
    "conversation": 45_000,
    "general": 25_000,
    "math": 70_000,
    "reasoning": 55_000,
    "coding": 70_000,
    "system_instruction": 40_000,
    "function_calling": 40_000,
    "rewriting": 25_000,
    "summarization": 30_000,
}


PATTERNS = {
    "math": [
        "solve",
        "equation",
        "calculate",
        "calculation",
        "derivative",
        "integral",
        "probability",
        "percentage",
        "algebra",
        "geometry",
        "arithmetic",
        "quadratic",
        "factor",
        "matrix",
        "vector",
        "statistics",
        "mean",
        "median",
        "variance",
        "standard deviation",
        "what is x",
        "find x",
    ],
    "coding": [
        "python",
        "javascript",
        "java",
        "c++",
        "c#",
        "code",
        "coding",
        "program",
        "function",
        "class",
        "bug",
        "debug",
        "error",
        "exception",
        "algorithm",
        "sql",
        "html",
        "css",
        "api",
        "github",
        "regex",
        "implement",
        "programming",
    ],
    "reasoning": [
        "reason",
        "reasoning",
        "explain why",
        "why",
        "logic",
        "logical",
        "step by step",
        "think",
        "deduce",
        "infer",
        "because",
        "therefore",
        "proof",
        "puzzle",
        "riddle",
        "compare",
        "analyze",
        "analysis",
    ],
    "system_instruction": [
        "system prompt",
        "system message",
        "instruction",
        "instructions",
        "you are an assistant",
        "act as",
        "role",
        "behavior",
        "follow these",
        "your task",
        "constraints",
        "guidelines",
    ],
    "function_calling": [
        "function call",
        "function calling",
        "tool call",
        "tool use",
        "tools",
        "api call",
        "arguments",
        "parameter",
        "parameters",
        "json",
        "schema",
        "function",
        "weather",
        "search",
        "database",
    ],
    "summarization": [
        "summarize",
        "summary",
        "summarization",
        "summarise",
        "shorten",
        "key points",
        "main points",
        "brief summary",
    ],
    "rewriting": [
        "rewrite",
        "rewritten",
        "rephrase",
        "paraphrase",
        "improve this",
        "make this",
        "polish",
        "edit this",
        "grammar",
        "correct this",
        "better wording",
        "professional version",
    ],
    "conversation": [
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "how are you",
        "good morning",
        "good evening",
        "bye",
        "conversation",
        "chat",
        "tell me about yourself",
        "what do you think",
    ],
}


def normalize_text(text):
    if text is None:
        return ""

    if isinstance(text, str):
        return text.lower()

    return str(text).lower()


def count_assistant_tokens(loss_mask):
    return sum(1 for x in loss_mask if int(x) != 0)


def validate_example(example):
    token_ids = example.get("token_ids")
    loss_mask = example.get("loss_mask")

    if not token_ids:
        return False

    if not loss_mask:
        return False

    if len(token_ids) != len(loss_mask):
        return False

    assistant_tokens = count_assistant_tokens(loss_mask)

    if assistant_tokens < MIN_ASSISTANT_TOKENS:
        return False

    return True


def classify_example(example):
    text = normalize_text(example.get("text", ""))

    messages = example.get("messages", [])

    if messages:
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "")
                content = message.get("content", "")

                text += " "
                text += normalize_text(role)
                text += " "
                text += normalize_text(content)

    categories = []

    for category, patterns in PATTERNS.items():
        for pattern in patterns:
            if pattern in text:
                categories.append(category)
                break

    if not categories:
        categories.append("general")

    if len(messages) >= 4:
        if "conversation" not in categories:
            categories.append("conversation")

    return categories


def create_chunks(token_ids, loss_mask):
    if len(token_ids) != len(loss_mask):
        return []

    if len(token_ids) <= MAX_CONTEXT_LENGTH:
        if count_assistant_tokens(loss_mask) >= MIN_ASSISTANT_TOKENS:
            return [
                {
                    "token_ids": list(token_ids),
                    "loss_mask": list(loss_mask),
                }
            ]

        return []

    chunks = []

    start = 0
    total_length = len(token_ids)

    while start < total_length:
        end = min(start + MAX_CONTEXT_LENGTH, total_length)

        chunk_tokens = token_ids[start:end]
        chunk_mask = loss_mask[start:end]

        assistant_tokens = count_assistant_tokens(chunk_mask)

        if assistant_tokens >= MIN_ASSISTANT_TOKENS:
            chunks.append(
                {
                    "token_ids": list(chunk_tokens),
                    "loss_mask": list(chunk_mask),
                }
            )

        if end >= total_length:
            break

        start += LONG_SEQUENCE_STRIDE

    return chunks


def choose_category(categories, selected_counts, targets):
    available = [
        category
        for category in categories
        if selected_counts[category] < targets[category]
    ]

    if not available:
        return None

    available.sort(
        key=lambda category: (
            selected_counts[category] / max(targets[category], 1),
            selected_counts[category],
        )
    )

    return available[0]


def print_progress(selected_count, category, selected_counts):
    print(
        f"Selected {selected_count:,} / {TARGET_TOTAL:,}"
    )

    print(f"  Current category: {category}")

    important_categories = [
        "math",
        "coding",
        "reasoning",
        "conversation",
        "function_calling",
    ]

    for name in important_categories:
        print(
            f"  {name.replace('_', ' ').title()}: "
            f"{selected_counts[name]:,}"
        )


def save_sharded_dataset(examples, output_dir, shard_size):
    os.makedirs(output_dir, exist_ok=True)

    shard_paths = []

    total = len(examples)

    for start in range(0, total, shard_size):
        end = min(start + shard_size, total)

        shard_examples = examples[start:end]

        dataset = Dataset.from_dict(
            {
                "token_ids": [
                    example["token_ids"]
                    for example in shard_examples
                ],
                "loss_mask": [
                    example["loss_mask"]
                    for example in shard_examples
                ],
                "category": [
                    example["category"]
                    for example in shard_examples
                ],
                "source_index": [
                    example["source_index"]
                    for example in shard_examples
                ],
            }
        )

        shard_name = f"shard_{start // shard_size:05d}"
        shard_path = os.path.join(output_dir, shard_name)

        dataset.save_to_disk(shard_path)

        shard_paths.append(
            {
                "name": shard_name,
                "path": shard_path,
                "examples": len(dataset),
            }
        )

        print(
            f"Saved {shard_name}: "
            f"{len(dataset):,} examples"
        )

    return shard_paths


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():
    print("=" * 80)
    print("K-GPT V2 SCALED DATASET BUILDER")
    print("=" * 80)

    random.seed(RANDOM_SEED)

    if os.path.exists(OUTPUT_DIR):
        print()
        print("Removing previous incomplete output...")
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("Loading cached SmolTalk dataset...")

    dataset = load_dataset(SOURCE_DATASET)

    train = dataset["train"]
    val = dataset["val"]

    print()
    print("Dataset loaded.")
    print(dataset)

    print()
    print(f"Training examples: {len(train):,}")
    print(f"Validation examples: {len(val):,}")

    print()
    print("=" * 80)
    print("BUILDING TRAINING CANDIDATES")
    print("=" * 80)

    indices = list(range(len(train)))
    random.shuffle(indices)

    selected_examples = []

    selected_counts = Counter()

    valid_original_examples = 0
    long_examples_chunked = 0
    generated_chunks = 0

    scanned = 0

    for index in indices:
        if len(selected_examples) >= TARGET_TOTAL:
            break

        example = train[index]

        if not validate_example(example):
            scanned += 1
            continue

        valid_original_examples += 1

        categories = classify_example(example)

        category = choose_category(
            categories,
            selected_counts,
            TARGETS,
        )

        if category is None:
            scanned += 1
            continue

        chunks = create_chunks(
            example["token_ids"],
            example["loss_mask"],
        )

        if len(chunks) == 0:
            scanned += 1
            continue

        if len(chunks) > 1:
            long_examples_chunked += 1

        generated_chunks += len(chunks)

        remaining = TARGET_TOTAL - len(selected_examples)

        chunks_to_add = chunks[:remaining]

        for chunk in chunks_to_add:
            selected_examples.append(
                {
                    "token_ids": chunk["token_ids"],
                    "loss_mask": chunk["loss_mask"],
                    "category": category,
                    "source_index": index,
                }
            )

            selected_counts[category] += 1

            if len(selected_examples) % 10_000 == 0:
                print_progress(
                    len(selected_examples),
                    category,
                    selected_counts,
                )

        scanned += 1

        if scanned % 100_000 == 0:
            print()
            print(
                f"Scanned {scanned:,} / {len(train):,}"
            )
            print(
                f"Current selected: "
                f"{len(selected_examples):,}"
            )

    print()
    print("=" * 80)
    print("TRAINING SELECTION COMPLETE")
    print("=" * 80)

    print(
        f"Selected: "
        f"{len(selected_examples):,}"
    )

    print(
        f"Valid original examples: "
        f"{valid_original_examples:,}"
    )

    print(
        f"Long examples chunked: "
        f"{long_examples_chunked:,}"
    )

    print(
        f"Generated chunks: "
        f"{generated_chunks:,}"
    )

    print()
    print("Category distribution:")

    for category, target in TARGETS.items():
        count = selected_counts[category]

        print(
            f"{category:<22} "
            f"{count:,} / {target:,}"
        )

    print()
    print("=" * 80)
    print("BUILDING VALIDATION DATASET")
    print("=" * 80)

    val_indices = list(range(len(val)))
    random.shuffle(val_indices)

    validation_examples = []

    validation_candidates = 0

    for index in val_indices:
        if len(validation_examples) >= VALIDATION_SIZE:
            break

        example = val[index]

        if not validate_example(example):
            continue

        chunks = create_chunks(
            example["token_ids"],
            example["loss_mask"],
        )

        if not chunks:
            continue

        validation_candidates += len(chunks)

        category_list = classify_example(example)

        category = (
            category_list[0]
            if category_list
            else "general"
        )

        for chunk in chunks:
            if len(validation_examples) >= VALIDATION_SIZE:
                break

            validation_examples.append(
                {
                    "token_ids": chunk["token_ids"],
                    "loss_mask": chunk["loss_mask"],
                    "category": category,
                    "source_index": index,
                }
            )

    print(
        f"Validation candidates: "
        f"{validation_candidates:,}"
    )

    print(
        f"Validation selected: "
        f"{len(validation_examples):,}"
    )

    print()
    print("=" * 80)
    print("SAVING TRAINING DATASET")
    print("=" * 80)

    train_output_dir = os.path.join(
        OUTPUT_DIR,
        "train",
    )

    val_output_dir = os.path.join(
        OUTPUT_DIR,
        "val",
    )

    train_shards = save_sharded_dataset(
        selected_examples,
        train_output_dir,
        TRAIN_SHARD_SIZE,
    )

    print()
    print("=" * 80)
    print("SAVING VALIDATION DATASET")
    print("=" * 80)

    val_shards = save_sharded_dataset(
        validation_examples,
        val_output_dir,
        VAL_SHARD_SIZE,
    )

    print()
    print("=" * 80)
    print("CALCULATING DATASET STATISTICS")
    print("=" * 80)

    train_token_count = 0
    train_assistant_token_count = 0

    train_lengths = []

    for example in selected_examples:
        token_count = len(example["token_ids"])
        assistant_count = count_assistant_tokens(
            example["loss_mask"]
        )

        train_token_count += token_count
        train_assistant_token_count += assistant_count

        train_lengths.append(token_count)

    val_token_count = 0
    val_assistant_token_count = 0

    val_lengths = []

    for example in validation_examples:
        token_count = len(example["token_ids"])
        assistant_count = count_assistant_tokens(
            example["loss_mask"]
        )

        val_token_count += token_count
        val_assistant_token_count += assistant_count

        val_lengths.append(token_count)

    train_lengths.sort()
    val_lengths.sort()

    def median(values):
        if not values:
            return 0

        n = len(values)
        middle = n // 2

        if n % 2 == 0:
            return (
                values[middle - 1]
                + values[middle]
            ) / 2

        return values[middle]

    report = {
        "source_dataset": SOURCE_DATASET,
        "random_seed": RANDOM_SEED,
        "max_context_length": MAX_CONTEXT_LENGTH,
        "min_assistant_tokens": MIN_ASSISTANT_TOKENS,
        "long_sequence_stride": LONG_SEQUENCE_STRIDE,
        "target_training_examples": TARGET_TOTAL,
        "actual_training_examples": len(selected_examples),
        "actual_validation_examples": len(validation_examples),
        "training": {
            "examples": len(selected_examples),
            "total_tokens": train_token_count,
            "assistant_loss_tokens": train_assistant_token_count,
            "assistant_token_percentage": (
                train_assistant_token_count
                / train_token_count
                * 100
                if train_token_count
                else 0
            ),
            "average_tokens": (
                train_token_count
                / len(selected_examples)
                if selected_examples
                else 0
            ),
            "median_tokens": median(train_lengths),
            "max_tokens": max(train_lengths)
            if train_lengths
            else 0,
            "min_tokens": min(train_lengths)
            if train_lengths
            else 0,
        },
        "validation": {
            "examples": len(validation_examples),
            "total_tokens": val_token_count,
            "assistant_loss_tokens": val_assistant_token_count,
            "assistant_token_percentage": (
                val_assistant_token_count
                / val_token_count
                * 100
                if val_token_count
                else 0
            ),
            "average_tokens": (
                val_token_count
                / len(validation_examples)
                if validation_examples
                else 0
            ),
            "median_tokens": median(val_lengths),
            "max_tokens": max(val_lengths)
            if val_lengths
            else 0,
            "min_tokens": min(val_lengths)
            if val_lengths
            else 0,
        },
        "category_distribution": dict(
            selected_counts
        ),
        "targets": TARGETS,
        "train_shards": train_shards,
        "validation_shards": val_shards,
    }

    report_path = os.path.join(
        OUTPUT_DIR,
        "dataset_report.json",
    )

    save_json(
        report_path,
        report,
    )

    manifest = {
        "dataset_type": "K-GPT V2 SFT",
        "format": "HuggingFace Dataset shards",
        "source_dataset": SOURCE_DATASET,
        "train_examples": len(selected_examples),
        "validation_examples": len(validation_examples),
        "max_context_length": MAX_CONTEXT_LENGTH,
        "train_shard_size": TRAIN_SHARD_SIZE,
        "validation_shard_size": VAL_SHARD_SIZE,
        "columns": [
            "token_ids",
            "loss_mask",
            "category",
            "source_index",
        ],
        "train_shards": [
            shard["name"]
            for shard in train_shards
        ],
        "validation_shards": [
            shard["name"]
            for shard in val_shards
        ],
    }

    manifest_path = os.path.join(
        OUTPUT_DIR,
        "manifest.json",
    )

    save_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 80)
    print("DATASET BUILD COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Training examples: "
        f"{len(selected_examples):,}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_examples):,}"
    )

    print(
        f"Training tokens: "
        f"{train_token_count:,}"
    )

    print(
        f"Assistant loss tokens: "
        f"{train_assistant_token_count:,}"
    )

    print(
        f"Average training sequence: "
        f"{train_token_count / len(selected_examples):.2f}"
        if selected_examples
        else "Average training sequence: 0"
    )

    print()
    print(
        "Training dataset:"
    )

    print(
        os.path.abspath(train_output_dir)
    )

    print()
    print(
        "Validation dataset:"
    )

    print(
        os.path.abspath(val_output_dir)
    )

    print()
    print(
        "Report:"
    )

    print(
        os.path.abspath(report_path)
    )

    print()
    print(
        "Manifest:"
    )

    print(
        os.path.abspath(manifest_path)
    )

    print()
    print("=" * 80)
    print("READY FOR TRAINING")
    print("=" * 80)


if __name__ == "__main__":
    main()
