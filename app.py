import streamlit as st
import torch
import tiktoken
from huggingface_hub import hf_hub_download

from model.model import GPTModel
from model.generate import generate


st.set_page_config(
    page_title="K-GPT",
    page_icon=None,
    layout="centered",
)

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16,
}

MODEL_REPO = "kg5290/K-GPT-model"
MODEL_FILE = "gpt2-medium355M-sft.pth"


@st.cache_resource(show_spinner=False)
def load_model():
    checkpoint_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    model = GPTModel(BASE_CONFIG)

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    model.load_state_dict(state, assign=True)
    model.eval()

    return model


@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def format_prompt(instruction):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
        "\n\n### Response:\n"
    )


def generate_response(instruction, max_new_tokens, temperature):
    model = load_model()
    tokenizer = get_tokenizer()

    prompt = format_prompt(instruction)
    encoded = tokenizer.encode(
        prompt,
        allowed_special={"<|endoftext|>"},
    )

    encoded = encoded[-BASE_CONFIG["context_length"]:]
    input_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    with torch.inference_mode():
        output_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            context_size=BASE_CONFIG["context_length"],
            temperature=temperature,
            eos_id=50256,
        )

    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())

    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):]

    if "### Response:" in output_text:
        output_text = output_text.split("### Response:", 1)[-1]

    if "### Instruction:" in output_text:
        output_text = output_text.split("### Instruction:", 1)[0]

    return output_text.strip()


st.title("K-GPT")
st.caption("Instruction-tuned GPT-2 Medium, fine-tuned for instruction following.")

instruction = st.text_area(
    "Instruction",
    placeholder="Enter an instruction for K-GPT...",
    height=150,
)

with st.expander("Generation settings"):
    max_new_tokens = st.slider(
        "Maximum new tokens",
        min_value=32,
        max_value=160,
        value=96,
        step=16,
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0 uses deterministic greedy generation. Higher values introduce more variation.",
    )

generate_clicked = st.button("Generate", type="primary", use_container_width=True)

if generate_clicked:
    if not instruction.strip():
        st.warning("Enter an instruction first.")
    else:
        with st.spinner("Generating response..."):
            try:
                response = generate_response(
                    instruction.strip(),
                    max_new_tokens,
                    temperature,
                )
                st.subheader("Response")
                st.write(response if response else "No response was generated.")
            except Exception as exc:
                st.error(
                    "The model could not be loaded or the response could not be generated."
                )
                st.exception(exc)

st.divider()
st.caption("K-GPT • GPT-2 Medium (355M parameters) • Local model inference")
