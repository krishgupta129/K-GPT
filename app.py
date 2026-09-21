import gc

import streamlit as st
import torch
from huggingface_hub import hf_hub_download

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

REPO_ID = "kg5290/K-GPT-model"
MODEL_FILE = "gpt2-medium355M-sft.pth"

MAX_CONTEXT_TOKENS = 864
DEFAULT_MAX_NEW_TOKENS = 160


st.set_page_config(
    page_title="K-GPT",
    page_icon=None,
    layout="centered",
)


def build_instruction_prompt(instruction, history):
    history_text = ""

    if history:
        history_lines = []

        for message in history:
            role = "User" if message["role"] == "user" else "K-GPT"
            history_lines.append(
                f"{role}: {message['content']}"
            )

        history_text = (
            "\n\n### Input:\n"
            "Previous conversation:\n"
            + "\n\n".join(history_lines)
        )

    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
        f"{history_text}"
        "\n\n### Response:\n"
    )


def trim_history_for_context(instruction, history):
    usable_history = list(history)

    while usable_history:
        prompt = build_instruction_prompt(
            instruction,
            usable_history,
        )

        if len(tokenizer.encode(prompt)) <= MAX_CONTEXT_TOKENS:
            return prompt

        usable_history.pop(0)

    prompt = build_instruction_prompt(
        instruction,
        [],
    )

    tokens = tokenizer.encode(prompt)

    if len(tokens) > MAX_CONTEXT_TOKENS:
        tokens = tokens[-MAX_CONTEXT_TOKENS:]
        prompt = tokenizer.decode(tokens)

    return prompt


@st.cache_resource(show_spinner="Loading K-GPT model...")
def load_model():
    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE,
    )

    with torch.device("meta"):
        model = GPTModel(BASE_CONFIG)

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state,
        assign=True,
    )

    del state
    gc.collect()

    model.eval()

    return model


def generate_response(instruction, history, max_new_tokens):
    model = load_model()

    prompt = trim_history_for_context(
        instruction,
        history,
    )

    token_ids = text_to_token_ids(
        prompt,
        tokenizer,
    )

    with torch.inference_mode():
        output_ids = generate(
            model=model,
            idx=token_ids,
            max_new_tokens=max_new_tokens,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )

    generated_text = token_ids_to_text(
        output_ids,
        tokenizer,
    )

    response = generated_text[len(prompt):].strip()

    if "### Response:" in response:
        response = response.replace(
            "### Response:",
            "",
            1,
        ).strip()

    return response


if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("K-GPT")
st.caption("GPT-2 Medium instruction-fine-tuned model.")


with st.sidebar:
    st.subheader("K-GPT")
    st.caption("Current session")

    if st.button(
        "New chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("Generation")

    max_new_tokens = st.slider(
        "Maximum response length",
        min_value=32,
        max_value=256,
        value=DEFAULT_MAX_NEW_TOKENS,
        step=16,
    )

    st.caption(
        "Conversation memory exists only for the current session."
    )


chat_height = (
    520
    if st.session_state.messages
    else 180
)


with st.container(
    height=chat_height,
    border=False,
):
    if not st.session_state.messages:
        st.caption(
            "Start a conversation below."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


prompt = st.chat_input("Message K-GPT")


if prompt:
    previous_history = list(
        st.session_state.messages
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.spinner("Generating response..."):
        try:
            response = generate_response(
                instruction=prompt,
                history=previous_history,
                max_new_tokens=max_new_tokens,
            )

        except Exception as exc:
            st.session_state.messages.pop()
            st.error(
                f"Unable to generate a response: {exc}"
            )
            st.stop()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    st.rerun()


st.caption(
    "K-GPT • GPT-2 Medium (355M parameters) • Session-based chat"
)
