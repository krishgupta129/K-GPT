import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False
    ):

        super().__init__()

        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    d_in,
                    d_out,
                    context_length,
                    dropout,
                    qkv_bias
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):

        return torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1
        )


class CausalAttention(nn.Module):

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        qkv_bias=False
    ):

        super().__init__()

        self.d_out = d_out

        self.W_query = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_key = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_value = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    context_length,
                    context_length
                ),
                diagonal=1
            )
        )

    def forward(self, x):

        queries = self.W_query(x)

        keys = self.W_key(x)

        values = self.W_value(x)

        attn_scores = (
            queries
            @ keys.transpose(
                1,
                2
            )
        )

        mask_bool = (
            self.mask.bool()
            [
                :x.shape[1],
                :x.shape[1]
            ]
        )

        attn_scores.masked_fill_(
            mask_bool,
            -torch.inf
        )

        attn_weights = torch.softmax(
            attn_scores
            / keys.shape[-1] ** 0.5,
            dim=-1
        )

        attn_weights = self.dropout(
            attn_weights
        )

        context_vec = (
            attn_weights
            @ values
        )

        return context_vec


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        num_heads,
        dropout,
        qkv_bias=False
    ):

        super().__init__()

        assert (
            d_out % num_heads == 0
        ), "d_out must be divisible by n_heads"

        self.d_out = d_out

        self.num_heads = num_heads

        self.head_dim = (
            d_out // num_heads
        )

        self.W_query = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_key = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_value = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.out_proj = nn.Linear(
            d_out,
            d_out
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    context_length,
                    context_length
                ),
                diagonal=1
            )
        )

    def forward(self, x):

        batch_size, num_tokens, _ = (
            x.shape
        )

        keys = self.W_key(x)

        queries = self.W_query(x)

        values = self.W_value(x)

        keys = keys.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        queries = queries.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        values = values.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        keys = keys.transpose(
            1,
            2
        )

        queries = queries.transpose(
            1,
            2
        )

        values = values.transpose(
            1,
            2
        )

        dropout_probability = (
            self.dropout.p
            if self.training
            else 0.0
        )

        context_vec = (
            F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=dropout_probability,
                is_causal=True
            )
        )

        context_vec = (
            context_vec.transpose(
                1,
                2
            )
        )

        context_vec = context_vec.reshape(
            batch_size,
            num_tokens,
            self.d_out
        )

        context_vec = self.out_proj(
            context_vec
        )

        return context_vec


class LayerNorm(nn.Module):

    def __init__(
        self,
        emb_dim
    ):

        super().__init__()

        self.eps = 1e-5

        self.scale = nn.Parameter(
            torch.ones(emb_dim)
        )

        self.shift = nn.Parameter(
            torch.zeros(emb_dim)
        )

    def forward(self, x):

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        var = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )

        norm_x = (
            x - mean
        ) / torch.sqrt(
            var + self.eps
        )

        return (
            self.scale
            * norm_x
            + self.shift
        )


class GELU(nn.Module):

    def __init__(self):

        super().__init__()

        self.constant = math.sqrt(
            2.0 / math.pi
        )

    def forward(self, x):

        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    self.constant
                    * (
                        x
                        + 0.044715
                        * torch.pow(
                            x,
                            3
                        )
                    )
                )
            )
        )


class FeedForward(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                cfg["emb_dim"],
                4 * cfg["emb_dim"]
            ),
            GELU(),
            nn.Linear(
                4 * cfg["emb_dim"],
                cfg["emb_dim"]
            )
        )

    def forward(self, x):

        return self.layers(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg[
                "context_length"
            ],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff = FeedForward(
            cfg
        )

        self.norm1 = LayerNorm(
            cfg["emb_dim"]
        )

        self.norm2 = LayerNorm(
            cfg["emb_dim"]
        )

        self.drop_resid = nn.Dropout(
            cfg["drop_rate"]
        )

    def forward(self, x):

        shortcut = x

        x = self.norm1(x)

        x = self.att(x)

        x = self.drop_resid(x)

        x = x + shortcut

        shortcut = x

        x = self.norm2(x)

        x = self.ff(x)

        x = self.drop_resid(x)

        x = x + shortcut

        return x


class GPTModel(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"]
        )

        self.pos_emb = nn.Embedding(
            cfg["context_length"],
            cfg["emb_dim"]
        )

        self.drop_emb = nn.Dropout(
            cfg["drop_rate"]
        )

        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(cfg)
                for _ in range(
                    cfg["n_layers"]
                )
            ]
        )

        self.final_norm = LayerNorm(
            cfg["emb_dim"]
        )

        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False
        )

    def forward(
        self,
        in_idx
    ):

        batch_size, seq_len = (
            in_idx.shape
        )

        tok_embeds = self.tok_emb(
            in_idx
        )

        pos_embeds = self.pos_emb(
            torch.arange(
                seq_len,
                device=in_idx.device
            )
        )

        x = (
            tok_embeds
            + pos_embeds
        )

        x = self.drop_emb(x)

        x = self.trf_blocks(x)

        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits
