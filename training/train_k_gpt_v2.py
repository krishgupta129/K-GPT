import os
import sys
import json
import math
import time
import argparse
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from model.model import GPTModel


BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16,
}


TRAIN_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "training",
    "k_gpt_v2_dataset",
    "train",
)

VAL_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "training",
    "k_gpt_v2_dataset",
    "val",
)

BASE_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "model",
    "weights",
    "gpt2-medium355M-sft.pth",
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "model",
    "checkpoints",
    "k_gpt_v2",
)

LATEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "latest.pth",
)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best.pth",
)


PAD_TOKEN_ID = 50256

BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 16

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.1

WARMUP_STEPS = 500

MAX_TRAIN_STEPS = 25000

VALIDATION_EVERY = 500

SAVE_EVERY = 500

MAX_GRAD_NORM = 1.0

NUM_WORKERS = 0

SEED = 42

SMOKE_TEST_STEPS = 20


class ShardedKGPTDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        shard_paths = sorted(
            glob(
                os.path.join(
                    root_dir,
                    "shard_*"
                )
            )
        )

        if not shard_paths:
            raise RuntimeError(
                f"No dataset shards found in:\n{root_dir}"
            )

        self.shards = []

        self.lengths = []

        for shard_path in shard_paths:

            dataset = load_from_disk(
                shard_path
            )

            self.shards.append(
                dataset
            )

            self.lengths.append(
                len(dataset)
            )

        self.cumulative_lengths = []

        total = 0

        for length in self.lengths:

            total += length

            self.cumulative_lengths.append(
                total
            )

        self.total_length = total

        print()
        print(
            f"Loaded {len(self.shards)} shards "
            f"from {root_dir}"
        )

        print(
            f"Total examples: "
            f"{self.total_length:,}"
        )

    def __len__(self):

        return self.total_length

    def __getitem__(self, index):

        if index < 0:

            index += self.total_length

        previous_total = 0

        for shard_index, cumulative_length in enumerate(
            self.cumulative_lengths
        ):

            if index < cumulative_length:

                local_index = (
                    index
                    - previous_total
                )

                example = self.shards[
                    shard_index
                ][local_index]

                return {
                    "token_ids": example[
                        "token_ids"
                    ],
                    "loss_mask": example[
                        "loss_mask"
                    ],
                }

            previous_total = cumulative_length

        raise IndexError(
            "Dataset index out of range."
        )


def collate_batch(batch):

    max_length = max(
        len(item["token_ids"])
        for item in batch
    )

    input_ids = []

    labels = []

    loss_masks = []

    for item in batch:

        tokens = item["token_ids"]

        mask = item["loss_mask"]

        padded_tokens = (
            tokens
            + [PAD_TOKEN_ID]
            * (
                max_length
                - len(tokens)
            )
        )

        padded_mask = (
            mask
            + [0]
            * (
                max_length
                - len(mask)
            )
        )

        input_sequence = (
            padded_tokens[:-1]
        )

        target_sequence = (
            padded_tokens[1:]
        )

        target_mask = (
            padded_mask[1:]
        )

        label_sequence = []

        for token, mask_value in zip(
            target_sequence,
            target_mask
        ):

            if mask_value:

                label_sequence.append(
                    token
                )

            else:

                label_sequence.append(
                    -100
                )

        input_ids.append(
            input_sequence
        )

        labels.append(
            label_sequence
        )

        loss_masks.append(
            target_mask
        )

    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.long
        ),
        "loss_mask": torch.tensor(
            loss_masks,
            dtype=torch.float32
        )
    }


def load_model(device):

    print()
    print("=" * 80)
    print("LOADING GPT-2 MEDIUM")
    print("=" * 80)

    if not os.path.exists(
        BASE_CHECKPOINT
    ):

        raise FileNotFoundError(
            f"Base checkpoint not found:\n"
            f"{BASE_CHECKPOINT}"
        )

    model = GPTModel(
        BASE_CONFIG
    )

    print()
    print(
        "Loading base checkpoint:"
    )

    print(
        BASE_CHECKPOINT
    )

    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu"
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model" in checkpoint:

            state_dict = checkpoint[
                "model"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    result = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if result.missing_keys:

        print()

        print(
            "Missing keys:"
        )

        for key in result.missing_keys[:10]:

            print(
                f"  {key}"
            )

    if result.unexpected_keys:

        print()

        print(
            "Unexpected keys:"
        )

        for key in result.unexpected_keys[:10]:

            print(
                f"  {key}"
            )

    model.to(device)

    return model


def count_parameters(model):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def calculate_loss(
    logits,
    labels
):

    batch_size = logits.shape[0]

    sequence_length = logits.shape[1]

    vocab_size = logits.shape[2]

    logits = logits.reshape(
        batch_size
        * sequence_length,
        vocab_size
    )

    labels = labels.reshape(
        batch_size
        * sequence_length
    )

    return nn.functional.cross_entropy(
        logits,
        labels,
        ignore_index=-100
    )


def get_learning_rate(step):

    if step < WARMUP_STEPS:

        return (
            LEARNING_RATE
            * (
                (step + 1)
                / WARMUP_STEPS
            )
        )

    progress = (
        step - WARMUP_STEPS
    ) / max(
        1,
        MAX_TRAIN_STEPS
        - WARMUP_STEPS
    )

    progress = min(
        max(
            progress,
            0.0
        ),
        1.0
    )

    cosine_decay = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )

    return (
        LEARNING_RATE
        * cosine_decay
    )


def set_learning_rate(
    optimizer,
    learning_rate
):

    for parameter_group in optimizer.param_groups:

        parameter_group[
            "lr"
        ] = learning_rate


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    step,
    best_val_loss
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    checkpoint = {
        "step": step,
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "best_val_loss":
            best_val_loss,
        "config":
            BASE_CONFIG
    }

    if scaler is not None:

        checkpoint[
            "scaler_state_dict"
        ] = scaler.state_dict()

    temporary_path = (
        path
        + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path
    )

    os.replace(
        temporary_path,
        path
    )


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    device,
    autocast_enabled
):

    model.eval()

    total_loss = 0.0

    total_batches = 0

    for batch in val_loader:

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True
        )

        labels = batch[
            "labels"
        ].to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=autocast_enabled
        ):

            logits = model(
                input_ids
            )

            loss = calculate_loss(
                logits,
                labels
            )

        total_loss += loss.item()

        total_batches += 1

    model.train()

    if total_batches == 0:

        return float("inf")

    return (
        total_loss
        / total_batches
    )


def save_training_config():

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    config = {
        "model_config":
            BASE_CONFIG,
        "base_checkpoint":
            BASE_CHECKPOINT,
        "train_data":
            TRAIN_DATA_DIR,
        "validation_data":
            VAL_DATA_DIR,
        "batch_size":
            BATCH_SIZE,
        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "warmup_steps":
            WARMUP_STEPS,
        "max_train_steps":
            MAX_TRAIN_STEPS,
        "validation_every":
            VALIDATION_EVERY,
        "save_every":
            SAVE_EVERY,
        "max_grad_norm":
            MAX_GRAD_NORM,
        "seed":
            SEED
    }

    config_path = os.path.join(
        CHECKPOINT_DIR,
        "training_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=2
        )


def train(
    smoke_test=False,
    resume=False
):

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )

        torch.set_float32_matmul_precision(
            "high"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 80)
    print("K-GPT V2 TRAINING")
    print("=" * 80)

    print()
    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        gpu_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU memory: "
            f"{gpu_memory:.2f} GB"
        )

    else:

        print()
        print(
            "WARNING: CUDA is not available."
        )

        print(
            "Training will be extremely slow."
        )

    save_training_config()

    train_dataset = (
        ShardedKGPTDataset(
            TRAIN_DATA_DIR
        )
    )

    val_dataset = (
        ShardedKGPTDataset(
            VAL_DATA_DIR
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch
    )

    model = load_model(
        device
    )

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    print()
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    autocast_enabled = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=autocast_enabled
    )

    start_step = 0

    best_val_loss = float(
        "inf"
    )

    if resume and os.path.exists(
        LATEST_CHECKPOINT
    ):

        print()
        print(
            "Resuming from:"
        )

        print(
            LATEST_CHECKPOINT
        )

        checkpoint = torch.load(
            LATEST_CHECKPOINT,
            map_location="cpu"
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if (
            autocast_enabled
            and "scaler_state_dict"
            in checkpoint
        ):

            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        start_step = checkpoint.get(
            "step",
            0
        )

        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf")
        )

        print(
            f"Resumed at step "
            f"{start_step:,}"
        )

    if smoke_test:

        target_steps = (
            SMOKE_TEST_STEPS
        )

    else:

        target_steps = (
            MAX_TRAIN_STEPS
        )

    print()
    print(
        f"Target steps: "
        f"{target_steps:,}"
    )

    print(
        f"Micro batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Effective batch size: "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print()
    print(
        "Starting training..."
    )

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    data_iterator = iter(
        train_loader
    )

    running_loss = 0.0

    running_batches = 0

    start_time = time.time()

    step = start_step

    while step < target_steps:

        step_start = time.time()

        accumulated_loss = 0.0

        for _ in range(
            GRADIENT_ACCUMULATION_STEPS
        ):

            try:

                batch = next(
                    data_iterator
                )

            except StopIteration:

                data_iterator = iter(
                    train_loader
                )

                batch = next(
                    data_iterator
                )

            input_ids = batch[
                "input_ids"
            ].to(
                device,
                non_blocking=True
            )

            labels = batch[
                "labels"
            ].to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=autocast_enabled
            ):

                logits = model(
                    input_ids
                )

                loss = calculate_loss(
                    logits,
                    labels
                )

                loss = (
                    loss
                    / GRADIENT_ACCUMULATION_STEPS
                )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite loss detected."
                )

            scaler.scale(
                loss
            ).backward()

            accumulated_loss += (
                loss.item()
                * GRADIENT_ACCUMULATION_STEPS
            )

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM
        )

        learning_rate = (
            get_learning_rate(
                step
            )
        )

        set_learning_rate(
            optimizer,
            learning_rate
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        optimizer.zero_grad(
            set_to_none=True
        )

        step += 1

        running_loss += (
            accumulated_loss
        )

        running_batches += 1

        step_time = (
            time.time()
            - step_start
        )

        if (
            step % 10 == 0
            or step == 1
        ):

            average_loss = (
                running_loss
                / running_batches
            )

            elapsed = (
                time.time()
                - start_time
            )

            tokens_per_step = (
                BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
                * BASE_CONFIG[
                    "context_length"
                ]
            )

            tokens_per_second = (
                tokens_per_step
                / max(
                    step_time,
                    1e-6
                )
            )

            print(
                f"Step "
                f"{step:>6,} / "
                f"{target_steps:,} | "
                f"Loss "
                f"{average_loss:.4f} | "
                f"LR "
                f"{learning_rate:.2e} | "
                f"{tokens_per_second:,.0f} tok/s | "
                f"{elapsed / 60:.1f} min"
            )

            running_loss = 0.0

            running_batches = 0

        if (
            step % VALIDATION_EVERY == 0
            or step == target_steps
        ):

            print()
            print("=" * 80)

            print(
                f"VALIDATION AT STEP "
                f"{step:,}"
            )

            val_loss = evaluate(
                model,
                val_loader,
                device,
                autocast_enabled
            )

            perplexity = math.exp(
                min(
                    val_loss,
                    20
                )
            )

            print(
                f"Validation loss: "
                f"{val_loss:.4f}"
            )

            print(
                f"Validation perplexity: "
                f"{perplexity:.2f}"
            )

            print("=" * 80)

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                save_checkpoint(
                    BEST_CHECKPOINT,
                    model,
                    optimizer,
                    scaler,
                    step,
                    best_val_loss
                )

                print()
                print(
                    "New best checkpoint saved."
                )

        if (
            step % SAVE_EVERY == 0
            or step == target_steps
        ):

            save_checkpoint(
                LATEST_CHECKPOINT,
                model,
                optimizer,
                scaler,
                step,
                best_val_loss
            )

            step_checkpoint = os.path.join(
                CHECKPOINT_DIR,
                f"step_{step:06d}.pth"
            )

            save_checkpoint(
                step_checkpoint,
                model,
                optimizer,
                scaler,
                step,
                best_val_loss
            )

            print()
            print(
                "Checkpoint saved:"
            )

            print(
                step_checkpoint
            )

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Final step: "
        f"{step:,}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        os.path.abspath(
            BEST_CHECKPOINT
        )
    )

    print()
    print(
        "Latest checkpoint:"
    )

    print(
        os.path.abspath(
            LATEST_CHECKPOINT
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small training test."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint."
    )

    args = parser.parse_args()

    train(
        smoke_test=args.smoke_test,
        resume=args.resume
    )


if __name__ == "__main__":

    main()
