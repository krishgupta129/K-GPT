SYSTEM_INSTRUCTION = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)


def build_prompt(instruction):
    return (
        f"{SYSTEM_INSTRUCTION}"
        f"\n\n### Instruction:\n{instruction}"
        "\n\n### Response:\n"
    )


def build_chat_prompt(instruction, history):
    if not history:
        return build_prompt(instruction)

    history_lines = []

    for message in history:
        role = "User" if message["role"] == "user" else "K-GPT"
        history_lines.append(
            f"{role}: {message['content']}"
        )

    history_text = "\n\n".join(history_lines)

    return (
        f"{SYSTEM_INSTRUCTION}"
        "\n\n### Input:\n"
        f"Previous conversation:\n{history_text}"
        f"\n\nCurrent user request:\n{instruction}"
        "\n\n### Response:\n"
    )
