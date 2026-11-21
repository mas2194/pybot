"""A complete hook-I/O byte model built from the official Griffith library.

The application serializes a hook's inputs and outputs; Griffith owns the
Transformer, LUTs and their differentiable/discrete implementations. Embeddings,
normalization, attention softmax and the vocabulary head remain floating point.
``hard`` means hard Griffith LUT evaluation, not a wholly Boolean machine.

Nothing in this module imports or calls the teacher's ``bot`` package. Generation
uses only the supplied input string and model weights, without answer caches,
response templates, rule fallbacks or a correct-output prefix.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from griffith_logic.convert.byte_vocab import encode_bytes
from griffith_logic.layers.lut import LutLayer
from griffith_logic.models import LogicLadderGPT, LogicLadderGPTConfig


PAD = 0
BOS = 1
EOS = 2
SEP = 3
BYTE_OFFSET = 4
VOCAB_SIZE = 260
IGNORE_INDEX = -100
TOKENIZER_VERSION = "utf8-bytes-offset4-pad0-bos1-eos2-sep3-v1"
CHECKPOINT_VERSION = "pybot-moduleio-griffith-v1"
GRIFFITH_REVISION = "e0675f4e36596954aa424d522e0ee966bba9a3e1"


def _integer(name: str, value: Any, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def encode_text(text: str) -> list[int]:
    """Official Griffith UTF-8 encoding plus application control-token offsets."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [value + BYTE_OFFSET for value in encode_bytes(text)]


@dataclass(frozen=True)
class ModelConfig:
    max_seq_len: int = 1024
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 1
    mlp_ratio: int = 2
    qk_bits: int | None = None
    v_bits: int | None = None
    logic: bool = True

    def __post_init__(self) -> None:
        # Griffith's LadderAttention returns v_bits channels directly into the
        # d_model-wide residual connection; it has no value output projection.
        # Resolve omitted widths from d_model, including custom-width models.
        if self.qk_bits is None:
            object.__setattr__(self, "qk_bits", self.d_model)
        if self.v_bits is None:
            object.__setattr__(self, "v_bits", self.d_model)
        for name in ("max_seq_len", "d_model", "n_heads", "n_layers", "mlp_ratio",
                     "qk_bits", "v_bits"):
            _integer(name, getattr(self, name))
        if self.max_seq_len < 4 or self.d_model < 4:
            raise ValueError("max_seq_len and d_model must be at least 4")
        if not isinstance(self.logic, bool):
            raise TypeError("logic must be a Boolean")
        if any(value % self.n_heads for value in (self.d_model, self.qk_bits, self.v_bits)):
            raise ValueError("d_model, qk_bits and v_bits must be divisible by n_heads")
        if self.logic and self.v_bits != self.d_model:
            raise ValueError("logic attention requires v_bits == d_model for its residual connection")


class ExampleTooLongError(ValueError):
    """The full input and output cannot fit; no text has been truncated."""


def prepare_example(input_text: str, output_text: str, max_seq_len: int) -> dict:
    """Encode every input/output byte, supervising output bytes and true EOS only.

    The first answer byte is predicted from the input's SEP. Later teacher-forced
    predictions see earlier output bytes only. Full-sequence length, including
    EOS, must fit the declared context; overlength examples fail explicitly.
    """
    _integer("max_seq_len", max_seq_len, 4)
    prompt = [BOS] + encode_text(input_text) + [SEP]
    answer = encode_text(output_text) + [EOS]
    full = prompt + answer
    if len(full) > max_seq_len:
        raise ExampleTooLongError(
            f"complete example needs {len(full)} tokens "
            f"(input {len(prompt)}, output including EOS {len(answer)}), "
            f"but max_seq_len={max_seq_len}; no input or output was truncated"
        )
    return {
        "input_ids": full[:-1],
        "labels": [IGNORE_INDEX] * (len(prompt) - 1) + answer,
        "prompt_ids": prompt,
        "answer_ids": answer,
        "input_tokens": len(prompt),
        "output_tokens": len(answer),
        "total_tokens": len(full),
    }


@contextmanager
def griffith_mode(module: nn.Module, mode: str):
    """Use native deterministic soft, hard-ST, or discrete LUT execution.

    Dropout is zero and attention normalization is softmax in every supported
    configuration. WARP's default Gumbel-Sigmoid noise must additionally be
    disabled for deterministic soft evaluation. All caller modes are restored.
    """
    if mode not in ("soft", "hard", "ste"):
        raise ValueError("mode must be soft, hard or ste")
    states = [(child, child.training) for child in module.modules()]
    lut_states = [(child, child.selection, child.gumbel, child.gumbel_noise)
                  for child, _ in states if isinstance(child, LutLayer)]
    try:
        module.train(mode != "hard")
        for layer, _, _, _ in lut_states:
            layer.selection = "hard_st" if mode == "ste" else "soft"
            layer.gumbel = False
            layer.gumbel_noise = False
        yield
    finally:
        for child, training in states:
            child.training = training
        for layer, selection, gumbel, gumbel_noise in lut_states:
            layer.selection = selection
            layer.gumbel = gumbel
            layer.gumbel_noise = gumbel_noise


class ModuleIOModel(nn.Module):
    """Griffith's causal Transformer with a lossless byte I/O interface."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.network = LogicLadderGPT(LogicLadderGPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=c.max_seq_len,
            d_model=c.d_model,
            n_heads=c.n_heads,
            n_layers=c.n_layers,
            mlp_ratio=c.mlp_ratio,
            dropout=0.0,
            attention="logic_proj" if c.logic else "fp",
            ffn="logic" if c.logic else "fp",
            attention_kwargs=({
                "qk_bits": c.qk_bits,
                "v_bits": c.v_bits,
                "n_inputs": 2,
                "parameterization": "warp",
                "proj_depth": 1,
                "score": "fp_dot",
                "norm": "softmax",
                "gumbel_noise": False,
            } if c.logic else {}),
        ))
        for layer in self.network.modules():
            if isinstance(layer, LutLayer):
                layer.gumbel = False
                layer.gumbel_noise = False
                layer.selection = "soft"

    def forward(self, input_ids: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have nonempty shape [batch, time]")
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError("input_ids must contain integer token IDs")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ExampleTooLongError("input exceeds max_seq_len; context is never cropped")
        if mode is None:
            return self.network(input_ids)
        with griffith_mode(self, mode):
            return self.network(input_ids)

    @torch.no_grad()
    def generate(self, input_text: str, max_new_tokens: int | None = None,
                 mode: str = "hard") -> dict:
        """Greedily generate a complete output, reporting EOS or budget exhaustion.

        ``max_new_tokens`` includes the possible EOS token. Omitting it uses the
        unused context capacity. A requested budget that would exceed capacity
        is an error, not permission to crop the original input. ``token_ids``
        contains content bytes only; ``generated_tokens`` includes EOS if seen.
        """
        if mode not in ("hard", "soft"):
            raise ValueError("generation mode must be hard or soft")
        prefix = [BOS] + encode_text(input_text) + [SEP]
        capacity = self.config.max_seq_len
        remaining = capacity - len(prefix)
        from_context = max_new_tokens is None
        if max_new_tokens is None:
            max_new_tokens = remaining
        else:
            _integer("max_new_tokens", max_new_tokens, 0)
        if remaining < 1:
            raise ExampleTooLongError(
                f"input uses {len(prefix)} tokens with max_seq_len={capacity}; "
                "no output token fits and the input will not be cropped"
            )
        if max_new_tokens > remaining:
            raise ExampleTooLongError(
                f"input {len(prefix)} + requested output budget {max_new_tokens} "
                f"exceeds max_seq_len={capacity}; maximum output budget is {remaining}"
            )
        device = next(self.parameters()).device
        sequence = torch.tensor([prefix], dtype=torch.long, device=device)
        generated: list[int] = []
        eos_generated = False
        with griffith_mode(self, mode):
            for _ in range(max_new_tokens):
                scores = self.network(sequence)[:, -1, :].clone()
                scores[:, [PAD, BOS, SEP]] = -torch.inf
                token = scores.argmax(dim=-1)
                value = int(token.item())
                if value == EOS:
                    eos_generated = True
                    break
                generated.append(value)
                sequence = torch.cat((sequence, token[:, None]), dim=1)
        raw = bytes(value - BYTE_OFFSET for value in generated)
        try:
            text = raw.decode("utf-8", errors="strict")
            utf8_valid = True
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
            utf8_valid = False
        return {
            "text": text,
            "token_ids": generated,
            "stop_reason": ("eos" if eos_generated else
                            "context_limit" if from_context else "max_new_tokens"),
            "eos_generated": eos_generated,
            "utf8_valid": utf8_valid,
            "input_tokens": len(prefix),
            "generated_tokens": len(generated) + int(eos_generated),
            "output_bytes": len(raw),
            "max_new_tokens": max_new_tokens,
            "max_seq_len": capacity,
            "mode": mode,
        }


def save_checkpoint(model: ModuleIOModel, path, metadata: dict | None = None) -> Path:
    """Save tensors and primitive metadata; no teacher source or response cache."""
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
    try:
        version = importlib_metadata.version("griffith-logic")
    except importlib_metadata.PackageNotFoundError:
        version = "source-checkout"
    payload = {
        "format": CHECKPOINT_VERSION,
        "model_config": asdict(model.config),
        "tokenizer": TOKENIZER_VERSION,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": metadata,
        "griffith": {"package_version": version,
                     "source_revision_constraint": GRIFFITH_REVISION},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


def load_checkpoint(path, device: str = "cpu") -> tuple[ModuleIOModel, dict]:
    """Load safely without importing PyBot, then default to hard LUT evaluation."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_VERSION:
        raise ValueError("not a supported module-I/O Griffith checkpoint")
    if payload.get("tokenizer") != TOKENIZER_VERSION:
        raise ValueError("checkpoint tokenizer does not match the byte I/O contract")
    model = ModuleIOModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload.get("metadata", {})
