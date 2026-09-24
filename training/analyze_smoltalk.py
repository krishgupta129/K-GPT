from datasets import load_dataset
from collections import Counter
from statistics import mean, median
import json
import os
import re

DATASET_NAME = "juliannunezb/smoltalk-gpt2-sft"

OUTPUT_DIR = "training"
REPORT_PATH = os.path.join(OUTPUT_DIR, "smoltalk_analysis.json")

MAX_CONTEXT_LENGTH = 1024
LONG_SEQUENCE_THRESHOLD = 512

SAMPLE_LONGEST = 20


def get_message_text(messages):
    parts = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", ""))
        content = str(message.get("content", ""))

        parts.append(f"{role}: {content}")

    return "\n".join(parts)


def get_roles(messages):
    roles = []

    for message in messages:
        if isinstance(message, dict):
            roles.append(str(message.get("role", "")).lower())

    return roles


def contains_any(text, patterns):
    text = text.lower()

    for pattern in patterns:
        if pattern in text:
            return True

    return False


def classify_example(messages):
    text = get_message_text(messages).lower()

    categories = []

    math_patterns = [
        "solve",
        "equation",
        "derivative",
        "integral",
        "probability",
        "geometry",
        "algebra",
        "calculus",
        "function",
        "quadratic",
        "matrix",
        "fraction",
        "percentage",
        "math",
        "mathematical",
        "proof",
        "theorem",
        "prime number",
        "logarithm",
        "trigonometry",
    ]

    coding_patterns = [
        "python",
        "javascript",
        "java",
        "c++",
        "c#",
        "program",
        "code",
        "coding",
        "function",
        "algorithm",
        "debug",
        "bug",
        "implement",
        "sql",
        "html",
        "css",
        "api",
        "class ",
        "def ",
        "for loop",
        "while loop",
    ]

    reasoning_patterns = [
        "reason",
        "explain why",
        "step by step",
        "analyze",
        "logic",
        "logical",
        "deduce",
        "deduction",
        "infer",
        "inference",
        "prove",
        "compare",
        "evaluate",
        "why",
    ]

    system_patterns = [
        "system:",
        "you are an ai assistant",
        "your response should",
        "provide a concise",
        "follow these instructions",
        "respond with",
        "must contain",
        "must not contain",
        "exactly",
        "at least",
        "at most",
    ]

    rewrite_patterns = [
        "rewrite",
        "rephrase",
        "paraphrase",
        "make it concise",
        "make this shorter",
        "edit the following",
        "improve the wording",
        "grammar",
        "proofread",
    ]

    summary_patterns = [
        "summarize",
        "summary",
        "summarise",
        "key points",
        "main points",
        "brief overview",
        "in a few sentences",
    ]

    conversation_patterns = [
        "hello",
        "hi",
        "how are you",
        "what do you think",
        "tell me about yourself",
        "thanks",
        "thank you",
        "good morning",
        "good evening",
    ]

    function_calling_patterns = [
        "function call",
        "function calling",
        "tool call",
        "tool calling",
        "arguments",
        "parameters",
        "api call",
        "json",
        "call the function",
        "use the tool",
    ]

    if contains_any(text, math_patterns):
        categories.append("math")

    if contains_any(text, coding_patterns):
        categories.append("coding")

    if contains_any(text, reasoning_patterns):
        categories.append("reasoning")

    if contains_any(text, system_patterns):
        categories.append("system_instruction")

    if contains_any(text, rewrite_patterns):
        categories.append("rewriting")

    if contains_any(text, summary_patterns):
        categories.append("summarization")

    if contains_any(text, conversation_patterns):
        categories.append("conversation")

    if contains_any(text, function_calling_patterns):
        categories.append("function_calling")

    if not categories:
        categories.append("general")

    return categories


def analyze_split(dataset, split_name):
    print("\n" + "=" * 80)
    print(f"ANALYZING SPLIT: {split_name}")
    print("=" * 80)

    total_examples = len(dataset)

    token_lengths = []
    assistant_token_lengths = []

    category_counts = Counter()

    role_counts = Counter()

    system_examples = 0
    multi_turn_examples = 0
    empty_assistant_examples = 0
    malformed_examples = 0

    over_512 = 0
    over_1024 = 0

    longest_examples = []

    total_assistant_tokens = 0
    total_tokens = 0

    min_tokens = None
    max_tokens = 0

    for index, example in enumerate(dataset):
        token_ids = example.get("token_ids", [])
        loss_mask = example.get("loss_mask", [])
        messages = example.get("messages", [])

        token_count = len(token_ids)

        assistant_token_count = sum(
            1 for value in loss_mask if int(value) == 1
        )

        token_lengths.append(token_count)
        assistant_token_lengths.append(assistant_token_count)

        total_tokens += token_count
        total_assistant_tokens += assistant_token_count

        if min_tokens is None or token_count < min_tokens:
            min_tokens = token_count

        if token_count > max_tokens:
            max_tokens = token_count

        if token_count > LONG_SEQUENCE_THRESHOLD:
            over_512 += 1

        if token_count > MAX_CONTEXT_LENGTH:
            over_1024 += 1

        roles = get_roles(messages)

        for role in roles:
            role_counts[role] += 1

        if "system" in roles:
            system_examples += 1

        if len(messages) >= 4:
            multi_turn_examples += 1

        assistant_messages = [
            message
            for message in messages
            if isinstance(message, dict)
            and str(message.get("role", "")).lower() == "assistant"
        ]

        if not assistant_messages:
            empty_assistant_examples += 1
        else:
            has_content = any(
                str(message.get("content", "")).strip()
                for message in assistant_messages
            )

            if not has_content:
                empty_assistant_examples += 1

        if not isinstance(messages, list) or len(messages) == 0:
            malformed_examples += 1

        categories = classify_example(messages)

        for category in categories:
            category_counts[category] += 1

        if len(longest_examples) < SAMPLE_LONGEST:
            longest_examples.append(
                {
                    "index": index,
                    "tokens": token_count,
                    "assistant_tokens": assistant_token_count,
                    "categories": categories,
                }
            )
            longest_examples.sort(
                key=lambda item: item["tokens"],
                reverse=True
            )
        elif token_count > longest_examples[-1]["tokens"]:
            longest_examples[-1] = {
                "index": index,
                "tokens": token_count,
                "assistant_tokens": assistant_token_count,
                "categories": categories,
            }
            longest_examples.sort(
                key=lambda item: item["tokens"],
                reverse=True
            )

        if (index + 1) % 100000 == 0:
            print(f"Processed {index + 1:,} / {total_examples:,}")

    average_tokens = mean(token_lengths) if token_lengths else 0
    median_tokens = median(token_lengths) if token_lengths else 0

    average_assistant_tokens = (
        mean(assistant_token_lengths)
        if assistant_token_lengths
        else 0
    )

    assistant_percentage = (
        (total_assistant_tokens / total_tokens) * 100
        if total_tokens > 0
        else 0
    )

    over_512_percentage = (
        (over_512 / total_examples) * 100
        if total_examples > 0
        else 0
    )

    over_1024_percentage = (
        (over_1024 / total_examples) * 100
        if total_examples > 0
        else 0
    )

    system_percentage = (
        (system_examples / total_examples) * 100
        if total_examples > 0
        else 0
    )

    multi_turn_percentage = (
        (multi_turn_examples / total_examples) * 100
        if total_examples > 0
        else 0
    )

    report = {
        "split": split_name,
        "total_examples": total_examples,
        "token_statistics": {
            "minimum": min_tokens,
            "maximum": max_tokens,
            "average": average_tokens,
            "median": median_tokens,
            "over_512": over_512,
            "over_512_percentage": over_512_percentage,
            "over_1024": over_1024,
            "over_1024_percentage": over_1024_percentage,
        },
        "assistant_loss_statistics": {
            "total_tokens": total_tokens,
            "total_assistant_tokens": total_assistant_tokens,
            "average_assistant_tokens": average_assistant_tokens,
            "assistant_token_percentage": assistant_percentage,
        },
        "conversation_statistics": {
            "system_examples": system_examples,
            "system_percentage": system_percentage,
            "multi_turn_examples": multi_turn_examples,
            "multi_turn_percentage": multi_turn_percentage,
            "empty_assistant_examples": empty_assistant_examples,
            "malformed_examples": malformed_examples,
        },
        "role_counts": dict(role_counts),
        "category_counts": dict(category_counts),
        "longest_examples": longest_examples,
    }

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)

    print(f"Total examples:              {total_examples:,}")
    print(f"Minimum tokens:              {min_tokens:,}")
    print(f"Maximum tokens:              {max_tokens:,}")
    print(f"Average tokens:              {average_tokens:.2f}")
    print(f"Median tokens:               {median_tokens:.2f}")

    print()
    print(f"Examples > {LONG_SEQUENCE_THRESHOLD}:           {over_512:,}")
    print(f"Examples > {LONG_SEQUENCE_THRESHOLD}:           {over_512_percentage:.2f}%")

    print(f"Examples > {MAX_CONTEXT_LENGTH}:          {over_1024:,}")
    print(f"Examples > {MAX_CONTEXT_LENGTH}:          {over_1024_percentage:.2f}%")

    print()
    print(f"Total tokens:                {total_tokens:,}")
    print(f"Assistant-loss tokens:       {total_assistant_tokens:,}")
    print(f"Average assistant tokens:    {average_assistant_tokens:.2f}")
    print(f"Assistant token percentage:   {assistant_percentage:.2f}%")

    print()
    print(f"System examples:             {system_examples:,}")
    print(f"System percentage:           {system_percentage:.2f}%")
    print(f"Multi-turn examples:         {multi_turn_examples:,}")
    print(f"Multi-turn percentage:       {multi_turn_percentage:.2f}%")
    print(f"Empty assistant examples:    {empty_assistant_examples:,}")
    print(f"Malformed examples:          {malformed_examples:,}")

    print("\nROLE COUNTS")
    print("-" * 80)

    for role, count in sorted(
        role_counts.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{role:20} {count:,}")

    print("\nCATEGORY SIGNALS")
    print("-" * 80)

    for category, count in sorted(
        category_counts.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        percentage = (
            (count / total_examples) * 100
            if total_examples > 0
            else 0
        )

        print(
            f"{category:20} {count:,} "
            f"({percentage:.2f}%)"
        )

    print("\nLONGEST EXAMPLES")
    print("-" * 80)

    for item in longest_examples:
        print(
            f"Index={item['index']:,} | "
            f"Tokens={item['tokens']:,} | "
            f"Assistant={item['assistant_tokens']:,} | "
            f"Categories={', '.join(item['categories'])}"
        )

    return report


def main():
    print("=" * 80)
    print("K-GPT v2 SmolTalk Dataset Analyzer")
    print("=" * 80)

    print("\nLoading cached dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded.")
    print(dataset)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_report = {}

    for split_name, split in dataset.items():
        full_report[split_name] = analyze_split(
            split,
            split_name
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            full_report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(f"\nReport saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
