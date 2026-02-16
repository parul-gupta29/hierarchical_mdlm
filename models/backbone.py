"""DDiT backbone wrapper — loads the pretrained MDLM-small checkpoint.

The backbone is the frozen (or LoRA-adapted) Diffusion Transformer that maps
(noisy token indices, sigma) -> hidden representations.

This module re-exports the internal layers of the HuggingFace-hosted DIT so
that HierarchicalGenerator can:
  1.  inject hierarchy embeddings *after* the token embedding layer, and
  2.  replace the single output head with K per-level heads.
"""

import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import flash_attn
    import flash_attn.layers.rotary
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

import huggingface_hub
import omegaconf


# ---------------------------------------------------------------------------
#  Low-level helpers (matching upstream MDLM exactly)
# ---------------------------------------------------------------------------

def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


@torch.jit.script
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
#  Core layers
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_dim: int = 1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)
        return self.cos_cached, self.sin_cached


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps (sigma) into vectors."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        if seqlens is None:
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=qkv.device,
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0.0, causal=False
        )
        x = rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)
        x = bias_dropout_scale_fn(
            self.attn_out(x), None, gate_msa, x_skip, self.dropout
        )

        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )
        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """Upstream-compatible DIT so we can load HF weights."""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        self.vocab_size = vocab_size
        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = Rotary(config.model.hidden_size // config.model.n_heads)

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    config.model.hidden_size,
                    config.model.n_heads,
                    config.model.cond_dim,
                    dropout=config.model.dropout,
                )
                for _ in range(config.model.n_blocks)
            ]
        )
        self.output_layer = DDitFinalLayer(
            config.model.hidden_size, vocab_size, config.model.cond_dim
        )
        self.scale_by_sigma = config.model.scale_by_sigma

    def forward(self, indices, sigma):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c)
            x = self.output_layer(x, c)
        return x


# ---------------------------------------------------------------------------
#  DDiTBackbone: high-level wrapper used by HierarchicalGenerator
# ---------------------------------------------------------------------------

class DDiTBackbone(nn.Module):
    """Wraps the pretrained DIT to expose encoder-only hidden states.

    After loading, the original ``output_layer`` is detached so that
    HierarchicalGenerator can plug in its own multi-head output.

    Public API used by HierarchicalGenerator:
        hidden = backbone.encode(token_ids, sigma)
            -> (B, L, D) hidden states *before* the final projection
        backbone.hidden_size   — int
        backbone.cond_dim      — int
        backbone.vocab_embed   — the token embedding table
    """

    # Default config matching the pretrained ``kuleshov-group/mdlm-owt``
    DEFAULT_CONFIG = {
        "model": {
            "hidden_size": 768,
            "n_heads": 12,
            "n_blocks": 12,
            "cond_dim": 128,
            "dropout": 0.1,
            "scale_by_sigma": True,
        }
    }

    def __init__(
        self,
        pretrained_path: str = "kuleshov-group/mdlm-owt",
        vocab_size: int = 50258,
        config_overrides: dict | None = None,
    ):
        super().__init__()
        cfg = dict(self.DEFAULT_CONFIG)
        if config_overrides:
            cfg["model"].update(config_overrides)
        self._cfg = omegaconf.OmegaConf.create(cfg)
        self.hidden_size: int = self._cfg.model.hidden_size
        self.cond_dim: int = self._cfg.model.cond_dim

        # Build the DIT and load pretrained weights
        self.dit = DIT(self._cfg, vocab_size=vocab_size)
        if pretrained_path:
            self._load_pretrained(pretrained_path, vocab_size)

        # Keep a reference for convenience
        self.vocab_embed = self.dit.vocab_embed
        self.sigma_map = self.dit.sigma_map
        self.rotary_emb = self.dit.rotary_emb
        self.blocks = self.dit.blocks

        # We still keep output_layer for weight-loading but don't use it
        # during hierarchical forward; the generator has its own heads.
        self.original_output_layer = self.dit.output_layer

    def _load_pretrained(self, path: str, vocab_size: int):
        """Load weights from a HuggingFace-hosted MDLM checkpoint."""
        try:
            pretrained = DIT.from_pretrained(path)
            # The pretrained model may have vocab_size 50258 (50257 + mask token)
            missing, unexpected = self.dit.load_state_dict(
                pretrained.state_dict(), strict=False
            )
            if missing:
                print(f"[DDiTBackbone] Missing keys (will init randomly): {missing}")
            if unexpected:
                print(f"[DDiTBackbone] Unexpected keys (ignored): {unexpected}")
            del pretrained
        except Exception as e:
            print(f"[DDiTBackbone] Could not load pretrained weights from "
                  f"'{path}': {e}. Starting from random init.")

    # ---- public API -------------------------------------------------------

    def encode(self, indices: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Run the DDiT encoder blocks and return hidden states (B, L, D).

        Does NOT apply the final output projection — the caller is
        responsible for passing these hidden states to per-level heads.
        """
        x = self.vocab_embed(indices)
        return self.encode_from_embeddings(x, sigma)

    def encode_from_embeddings(
        self, x: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Same as ``encode`` but takes pre-computed embeddings (B, L, D).

        This is the hook that lets HierarchicalGenerator inject hierarchy
        embeddings before the transformer blocks.
        """
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c)
        return x

    def get_timestep_conditioning(self, sigma: torch.Tensor) -> torch.Tensor:
        """Return the conditioning vector for sigma (needed by output heads)."""
        return F.silu(self.sigma_map(sigma))
