# K-GPT

K-GPT is an instruction-tuned GPT-2 Medium language model built from the
GPT implementation developed while working through Sebastian Raschka's
*Build a Large Language Model (From Scratch)* material. 
A GPT-2 Medium model instruction-fine-tuned on instruction-response data, 
deployed with a session-based conversational interface.

## Deployment architecture

- **GitHub:** application source code and model implementation
- **Hugging Face:** 1.73 GB fine-tuned model checkpoint
- **Streamlit Community Cloud:** public web interface

The model checkpoint is intentionally kept outside the GitHub repository.

## Model

- Architecture: GPT-2 Medium
- Parameters: approximately 355M
- Vocabulary: 50,257 tokens
- Context length: 1,024 tokens
- Fine-tuned for instruction following

## Running locally

From the repository root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit app downloads the public checkpoint from the Hugging Face
model repository on first load and caches the loaded model for subsequent
requests.

## Notes

K-GPT is a personal/educational project. Its responses can be incorrect,
incomplete, or inconsistent on some tasks.
