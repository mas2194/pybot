"""Teacher-output-only training for complete, bounded module I/O strings.

Teacher-forced token accuracy is a diagnostic. It is deliberately labeled as
such and must not be described as free-generation or complete-hook accuracy.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import random
import time

import torch
from torch.nn import functional as F

from .model import (IGNORE_INDEX, PAD, ModelConfig, ModuleIOModel,
                    load_checkpoint, prepare_example, save_checkpoint)


def _prepare(examples: list[dict], config: ModelConfig) -> list[dict]:
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples must be a nonempty list")
    prepared = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise TypeError(f"example {index} must be a dictionary")
        if "input_text" not in example or "output_text" not in example:
            raise ValueError(f"example {index} requires input_text and output_text")
        try:
            prepared.append(prepare_example(example["input_text"], example["output_text"],
                                             config.max_seq_len))
        except ValueError as exc:
            raise ValueError(f"example {index} ({example.get('module', '?')}."
                             f"{example.get('hook', '?')}): {exc}") from exc
    return prepared


def collate(examples: list[dict], device: str | torch.device = "cpu") -> dict:
    """Right-pad inputs; causal masking makes future padding unavailable to loss."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    length = max(len(example["input_ids"]) for example in examples)
    inputs = torch.full((len(examples), length), PAD, dtype=torch.long, device=device)
    labels = torch.full_like(inputs, IGNORE_INDEX)
    for row, example in enumerate(examples):
        size = len(example["input_ids"])
        inputs[row, :size] = torch.tensor(example["input_ids"], device=device)
        labels[row, :size] = torch.tensor(example["labels"], device=device)
    return {"input_ids": inputs, "labels": labels}


@torch.no_grad()
def _metrics(model: ModuleIOModel, prepared: list[dict], mode: str, batch_size: int) -> dict:
    device = next(model.parameters()).device
    loss_sum = 0.0
    token_count = correct = sequence_correct = 0
    for start in range(0, len(prepared), batch_size):
        batch = collate(prepared[start:start + batch_size], device)
        logits = model(batch["input_ids"], mode=mode)
        labels = batch["labels"]
        mask = labels != IGNORE_INDEX
        loss_sum += float(F.cross_entropy(logits[mask], labels[mask], reduction="sum"))
        hits = (logits.argmax(-1) == labels) | ~mask
        token_count += int(mask.sum())
        correct += int((hits & mask).sum())
        sequence_correct += int(hits.all(dim=1).sum())
    nll = loss_sum / token_count
    return {
        "conditioning": "correct output prefix (teacher forcing); not free generation",
        "mode": mode,
        "examples": len(prepared),
        "output_tokens_including_eos": token_count,
        "output_nll_nats": nll,
        "output_bits_per_token": nll / math.log(2),
        "teacher_forced_token_accuracy": correct / token_count,
        "teacher_forced_sequence_accuracy": sequence_correct / len(prepared),
    }


def teacher_forced_metrics(model: ModuleIOModel, examples: list[dict], mode: str = "hard",
                           batch_size: int = 4) -> dict:
    if mode not in ("soft", "hard"):
        raise ValueError("evaluation mode must be soft or hard")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return _metrics(model, _prepare(examples, model.config), mode, batch_size)


def fit(examples: list[dict], config: ModelConfig | None = None, *,
        steps: int = 100, batch_size: int = 4, seed: int = 0,
        learning_rate: float = 0.003, threads: int = 1,
        train_mode: str = "soft", device: str = "cpu", progress=None
        ) -> tuple[ModuleIOModel, dict]:
    """Fit all declared output bytes; reject every overlength record explicitly.

    Callers supply only training examples. Split assignment, held-out evaluation
    and free-generation semantic comparison belong to the dataset/evaluation
    layers. This function never imports or executes the teacher.
    """
    config = config or ModelConfig()
    for name, value in (("steps", steps), ("batch_size", batch_size), ("threads", threads)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) \
            or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if train_mode not in ("soft", "ste"):
        raise ValueError("train_mode must be soft or ste")
    prepared = _prepare(examples, config)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = ModuleIOModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    before = {mode: _metrics(model, prepared, mode, batch_size) for mode in ("soft", "hard")}
    curve = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        chosen = [rng.choice(prepared) for _ in range(batch_size)]
        batch = collate(chosen, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], mode=train_mode)
        labels = batch["labels"]
        mask = labels != IGNORE_INDEX
        loss = F.cross_entropy(logits[mask], labels[mask])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss; checkpoint has not been saved")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("nonfinite training gradients; checkpoint has not been saved")
        optimizer.step()
        if step == 1 or step == steps or step % max(1, steps // 10) == 0:
            point = {"step": step, "output_nll_nats": float(loss.detach()),
                     "gradient_norm_before_clipping": float(gradient_norm),
                     "elapsed_seconds": time.perf_counter() - started}
            curve.append(point)
            if progress is not None:
                progress(point)
    elapsed = time.perf_counter() - started
    after = {mode: _metrics(model, prepared, mode, batch_size) for mode in ("soft", "hard")}
    source = [{"input_text": example["input_text"], "output_text": example["output_text"]}
              for example in examples]
    dataset_hash = hashlib.sha256(json.dumps(source, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    report = {
        "format": "moduleio-training-report-v1",
        "model": asdict(config),
        "training": {"steps": steps, "batch_size": batch_size, "seed": seed,
                     "learning_rate": learning_rate, "threads": threads,
                     "train_mode": train_mode, "device": str(device)},
        "training_examples": len(examples),
        "dataset_sha256": dataset_hash,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "max_input_tokens": max(example["input_tokens"] for example in prepared),
        "max_output_tokens_including_eos": max(example["output_tokens"] for example in prepared),
        "max_total_tokens": max(example["total_tokens"] for example in prepared),
        "truncated_examples": 0,
        "training_seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "teacher_forced_before": before,
        "teacher_forced_after": after,
        "curve": curve,
        "torch_version": str(torch.__version__),
        "implementation": "official Griffith LogicLadderGPT; FP embedding/head/normalization/softmax remain",
        "limitations": "Teacher-forced diagnostics do not establish complete free-generation I/O agreement.",
    }
    return model.eval(), report
