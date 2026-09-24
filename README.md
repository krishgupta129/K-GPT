# K-GPT

K-GPT is a locally developed instruction-tuned GPT language model built from a GPT implementation developed while working through Sebastian Raschka's *Build a Large Language Model (From Scratch)* material.

K-GPT v2 Beta is a continued supervised fine-tuning stage of the original K-GPT model, focused on improving instruction-following behavior using curated instruction-response data.

---

## K-GPT v2 Beta

The current Beta release is based on the model checkpoint saved at **training step 17,000**.

### Model Specifications

| Specification | Value |
|---|---|
| Architecture | GPT-2 Medium-style architecture |
| Trainable Parameters | 406,286,336 |
| Vocabulary Size | 50,257 tokens |
| Context Length | 1,024 tokens |
| Transformer Layers | 24 |
| Attention Heads | 16 |
| Embedding Dimension | 1,024 |
| Tokenizer | GPT-2 BPE |
| Training | Instruction-response supervised fine-tuning |
| Current Beta Checkpoint | Step 17,000 |

The GPT-2 Medium naming is retained for architectural reference. The implemented K-GPT model contains **406,286,336 trainable parameters**.

---

# Architecture

K-GPT uses a decoder-only Transformer architecture based on the GPT-2 family.

```text
                         K-GPT
                           │
                           ▼
                    GPT-2 Tokenizer
                           │
                           ▼
                     Token Embeddings
                           │
                           ▼
                  Transformer Blocks
                           │
              ┌────────────┴────────────┐
              │                         │
        Multi-Head Attention        Feed Forward
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                    Repeated 24×
                           │
                           ▼
                    Output Projection
                           │
                           ▼
                    Generated Tokens

The model uses:

24 Transformer layers
16 attention heads
1,024-dimensional embeddings
1,024-token context window
GPT-2 BPE vocabulary of 50,257 tokens
Causal self-attention
Instruction-response supervised fine-tuning
K-GPT v2 Beta Inference

K-GPT v2 Beta uses a universal instruction prompt template for inference.

Universal K-GPT Inference Template
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
<USER QUERY>

### Response:

The template is implemented centrally in:

model/prompt_template.py

This keeps the inference format consistent across local inference, conversational inference, and future deployment interfaces.

Inference Pipeline
User Query
    │
    ▼
Universal K-GPT Prompt Template
    │
    ▼
GPT-2 BPE Tokenizer
    │
    ▼
K-GPT v2 Beta
    │
    ▼
Autoregressive Token Generation
    │
    ▼
EOS / EOT Handling
    │
    ▼
Response Cleanup
    │
    ▼
User Response

The Universal Prompt Template is an inference-layer component. It does not modify the model weights.

Conversational Inference

K-GPT v2 Beta also supports a conversation-oriented prompt structure.

Previous conversation can be provided to the model together with the current request:

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Input:
Previous conversation:
User: ...
K-GPT: ...

Current user request:
...

### Response:

The chat template is implemented in:

model/prompt_template.py
Repository Structure
K-GPT/
│
├── app.py
├── run_local.py
├── run_local_chat.py
├── README.md
├── requirements.txt
│
├── model/
│   ├── model.py
│   ├── generate.py
│   ├── tokenizer.py
│   ├── prompt_template.py
│   │
│   ├── weights/
│   │   └── k_gpt_v2_beta_deploy.pth
│   │
│   └── checkpoints/
│       └── k_gpt_v2/
│
├── training/
│   └── train_k_gpt_v2.py
│
└── tests/
    └── test_k_gpt_v2_beta.py

Large model files, training checkpoints, datasets, credentials, and other private artifacts are intentionally excluded from the Git repository.

Main Components
model/model.py

Contains the K-GPT Transformer architecture.

The implementation is checkpoint-compatible with the K-GPT v2 training checkpoint and contains:

406,286,336 trainable parameters

The attention implementation uses PyTorch's scaled dot-product attention where supported.

model/generate.py

Contains autoregressive text generation utilities.

The generation pipeline supports:

Greedy generation
Temperature-based sampling
Top-k filtering
EOS token handling
Token/text conversion
Generated response cleanup

The GPT-2 end-of-text token is:

50256
model/tokenizer.py

K-GPT uses the GPT-2 BPE tokenizer through tiktoken.

Vocabulary size: 50,257
model/prompt_template.py

Contains the official K-GPT inference templates.

This file acts as the single source of truth for prompt construction.

run_local.py

Provides local single-turn inference.

Run:

python run_local.py

The local application automatically uses the K-GPT v2 Beta deployment weights.

run_local_chat.py

Provides local conversational inference.

Run:

python run_local_chat.py

Supported commands include:

/exit
/quit
/clear
app.py

Provides the Streamlit-based web interface.

The application is intended to provide a conversational interface for K-GPT.

Local Installation

Clone the repository:

git clone https://github.com/krishgupta129/K-GPT.git
cd K-GPT

Install the required dependencies:

pip install -r requirements.txt
Local Inference
Single-Turn Mode

Run:

python run_local.py

Example:

You: What is artificial intelligence?

K-GPT: ...

The application automatically applies the Universal K-GPT Inference Template before sending the request to the model.

Chat Mode

Run:

python run_local_chat.py

Example:

You: What is machine learning?

K-GPT: ...

You: Give me an example.

K-GPT: ...

Conversation history is incorporated into the chat prompt.

Streamlit

Run the local Streamlit application with:

streamlit run app.py

The public deployment architecture is designed around:

GitHub
   │
   │ source code
   ▼
Streamlit
   │
   ▼
K-GPT inference
Model Files

The K-GPT v2 Beta deployment model is:

model/weights/k_gpt_v2_beta_deploy.pth

The deployment weight contains the model state dictionary without the optimizer and training-state information required for resuming training.

The full Step 17,000 training checkpoint is maintained separately:

model/checkpoints/k_gpt_v2/step_017000.pth

The full training checkpoint contains additional training information such as:

Model state
Optimizer state
Gradient scaler state
Training step
Validation information
Model configuration

Large checkpoint files are intentionally excluded from GitHub.

Training

K-GPT v2 is a continued supervised fine-tuning stage of the original K-GPT model.

The training system supports:

GPU training
Gradient accumulation
Periodic validation
Checkpoint saving
Training resumption
Best-checkpoint tracking
Automatic checkpoint backups

The current Beta checkpoint was saved at:

Step 17,000

The training pipeline is located in:

training/
Dataset

K-GPT v2 training uses a curated instruction-response dataset prepared for the project.

The processed dataset is organized into training and validation shards.

The dataset itself is intentionally excluded from GitHub because of its size.

training/
└── k_gpt_v2_dataset/
    ├── train/
    └── val/

The local training dataset is not required for standard model inference.

Beta Evaluation

K-GPT v2 Beta includes a standardized evaluation script:

python -m tests.test_k_gpt_v2_beta

The evaluation set covers areas including:

General knowledge
Definitions
Mathematics
Programming
Simple reasoning
Machine learning concepts
Instruction following
Short-form generation
Repetition behavior
Factual questions

The current Beta is an active development checkpoint.

Some responses may be:

Incorrect
Repetitive
Incomplete
Inconsistent
Factually inaccurate
Weak on mathematical reasoning

These limitations are part of the current development stage and are targets for future model improvements.

K-GPT v1 → v2

K-GPT v2 is developed as a continuation of the original K-GPT instruction-tuned model.

Original K-GPT
      │
      ▼
Continued SFT
      │
      ▼
K-GPT v2 Training
      │
      ▼
Step 17,000
      │
      ▼
K-GPT v2 Beta
      │
      ▼
Future Training & Improvements

The original K-GPT v1 checkpoint remains preserved separately and is not overwritten by the v2 Beta release.

Deployment Architecture

K-GPT separates source code, deployable model weights, and training backups.

                    K-GPT Project
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
       GitHub        Hugging Face    Google Drive
          │              │              │
          ▼              ▼              ▼
     Source Code     Beta Model     Training
     Architecture    Weights       Checkpoints
     Tests            1.73 GB       Backups
     Documentation
GitHub

Contains:

Model architecture
Inference code
Prompt templates
Training source
Evaluation scripts
Documentation

Large model artifacts and private credentials are excluded.

Hugging Face

Used for hosting the deployable K-GPT v2 Beta model weights.

Google Drive

Used for large training checkpoints and backup files during model development.

Streamlit

Used for the public web interface.

Checkpoint Strategy

Training checkpoints are maintained separately from deployment weights.

Training Checkpoint
        │
        ├── Model State
        ├── Optimizer State
        ├── Scaler State
        ├── Training Step
        └── Configuration
                 │
                 ▼
        Deployment Extraction
                 │
                 ▼
      Model State Dictionary
                 │
                 ▼
     k_gpt_v2_beta_deploy.pth

This allows the same training checkpoint to be used for future training while keeping a smaller deployment-oriented model file for inference.

Current Beta Status
K-GPT v2 Beta
Training checkpoint: Step 17,000
Status: Active development

The current release establishes the foundation for future versions.

Potential future improvements include:

Better mathematical reliability
Improved factual accuracy
Better reasoning behavior
Reduced repetition
Stronger instruction following
Better conversational consistency
Additional supervised fine-tuning
Improved evaluation coverage
Improved inference strategies

The Beta is intended to be iterated upon rather than treated as a final model.

Future Roadmap
K-GPT v2 Beta
      │
      ├── Real-world testing
      │
      ├── Failure analysis
      │
      ├── Dataset improvements
      │
      ├── Additional training
      │
      ├── Improved reasoning
      │
      ├── Improved factual reliability
      │
      └── Future K-GPT releases

Future checkpoints may be released as new versions as training and evaluation progress.

Limitations

K-GPT is a personal/educational model project.

The model can generate incorrect, incomplete, repetitive, or inconsistent responses.

It should not be treated as an authoritative source for medical, legal, financial, scientific, or other high-stakes decisions.

The current Beta is primarily intended for experimentation, development, research, and demonstration.

Project Philosophy

K-GPT is developed incrementally.

The project prioritizes:

Building the model from understandable components
Maintaining reproducible training checkpoints
Keeping the inference pipeline transparent
Testing real model behavior
Improving the model through successive training iterations
Keeping the project locally controllable
Avoiding unnecessary dependence on paid external model APIs

The goal is not simply to wrap an existing API, but to understand and develop the complete model pipeline:

Dataset
   ↓
Tokenization
   ↓
Model Architecture
   ↓
Supervised Fine-Tuning
   ↓
Checkpointing
   ↓
Inference
   ↓
Prompt Engineering
   ↓
Evaluation
   ↓
Deployment
   ↓
Future Training
Author

Krish Gupta

K-GPT is an independent AI/ML development project focused on understanding and building language-model systems from the ground up.

Project Links

GitHub:

https://github.com/krishgupta129/K-GPT

Hugging Face:

https://huggingface.co/kg5290/K-GPT-model

Streamlit:

https://k-gpt-model.streamlit.app

License

This project is currently maintained as a personal/educational AI project.

See the repository for the latest project terms and usage information.
