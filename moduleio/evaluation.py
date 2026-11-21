"""Whole-call evaluation from input-only, freely generated model output.

No PyBot import or teacher call is made here.  Every attempted example stays in
the denominator, including context overflows and incomplete/invalid outputs.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
import time
from collections import defaultdict
from typing import Any, Callable, Iterable

from . import codec


class InvalidOutputError(ValueError):
    """A generated value does not satisfy the complete module I/O contract."""


def validate_output(value: Any, hook: str) -> dict:
    """Validate the envelope and put its fields in one fixed order.

    The order of the envelope itself is immaterial.  The order, Python types,
    and alias relationships *inside* the returned value and arguments remain
    meaningful and are compared by the lossless codec.
    """
    required = {"return", "memory", "exception"}
    if hook == "choose":
        required.add("candidates")
    if type(value) is not dict or set(value) != required:
        found = sorted(str(key) for key in value) if isinstance(value, dict) else type(value).__name__
        raise InvalidOutputError(f"Expected output fields {sorted(required)}, got {found}")
    if type(value["memory"]) is not dict:
        raise InvalidOutputError("memory must be the full updated dictionary")
    exception = value["exception"]
    if exception is not None:
        if (type(exception) is not dict or set(exception) != {"type", "args"}
                or not isinstance(exception["type"], str) or not exception["type"]
                or not isinstance(exception["args"], tuple)):
            raise InvalidOutputError("exception must be None or {type: nonempty str, args: tuple}")
    if hook == "choose" and type(value["candidates"]) is not list:
        raise InvalidOutputError("choose must return the updated candidates list")
    normalized = {"return": value["return"], "memory": value["memory"], "exception": exception}
    if hook == "choose":
        normalized["candidates"] = value["candidates"]
    return normalized


def decode_output(text: str, hook: str) -> dict:
    """Decode one complete typed JSON value, rejecting partial/excess fields."""
    return validate_output(codec.loads(text), hook)


def exact_equal(left: Any, right: Any) -> bool:
    """Equality of encoded values, including type, order, and shared references."""
    return codec.dumps(left) == codec.dumps(right)


def compare_outputs(expected: dict, actual: dict, hook: str) -> dict:
    expected = validate_output(expected, hook)
    actual = validate_output(actual, hook)
    fields = ("return", "memory", "exception") + (("candidates",) if hook == "choose" else ())
    result = {f"{name}_exact": exact_equal(expected[name], actual[name]) for name in fields}
    result["whole_call_exact"] = exact_equal(expected, actual)
    return result


def _arguments(example: dict) -> dict:
    record = example.get("record")
    if isinstance(record, dict) and "input" in record:
        return codec.decode(record["input"])
    if "arguments" in example:
        return example["arguments"]
    raise ValueError("An example needs its record.input or arguments to evaluate the pass baseline")


def _active(example: dict, expected: dict) -> bool:
    try:
        before = _arguments(example)
    except ValueError:
        if "active" not in example:
            raise ValueError("Activity requires arguments or an explicit active flag")
        return bool(example["active"])
    if expected["return"] is not None or expected["exception"] is not None:
        return True
    if not exact_equal(before["memory"], expected["memory"]):
        return True
    return example["hook"] == "choose" and not exact_equal(before["candidates"], expected["candidates"])


def _empty_row(example: dict, index: int, expected: dict) -> dict:
    return {
        "index": index,
        "group": example.get("group"),
        "module": example["module"],
        "hook": example["hook"],
        "active": _active(example, expected),
        "status": "generation_error",
        "whole_call_exact": False,
        "return_exact": False,
        "memory_exact": False,
        "exception_exact": False,
        "candidates_exact": False if example["hook"] == "choose" else None,
        "elapsed_seconds": 0.0,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summarize(rows: list[dict]) -> dict:
    total = len(rows)
    choose_total = sum(row["hook"] == "choose" for row in rows)
    fields = ("whole_call_exact", "return_exact", "memory_exact", "exception_exact", "candidates_exact")
    counts = {field: sum(row.get(field) is True for row in rows) for field in fields}
    statuses = ("ok", "invalid_json", "invalid_schema", "invalid_utf8", "eos_missing", "overflow", "generation_error")
    counts.update({status: sum(row["status"] == status for row in rows) for status in statuses})
    counts["examples"] = total
    counts["choose_examples"] = choose_total
    rates = {field: (counts[field] / total if total else None) for field in fields if field != "candidates_exact"}
    rates["candidates_exact"] = counts["candidates_exact"] / choose_total if choose_total else None
    rates["valid_complete_output"] = counts["ok"] / total if total else None
    elapsed = [row["elapsed_seconds"] for row in rows]
    return {
        "counts": counts,
        "rates": rates,
        "latency_seconds": {
            "total": sum(elapsed),
            "mean": statistics.mean(elapsed) if elapsed else None,
            "median": statistics.median(elapsed) if elapsed else None,
            "p95": _percentile(elapsed, 0.95),
            "max": max(elapsed) if elapsed else None,
        },
    }


def _report(rows: list[dict], mode: str, method: str) -> dict:
    report = {"method": method, "mode": mode, **_summarize(rows)}
    report["by_activity"] = {
        "active": _summarize([row for row in rows if row["active"]]),
        "inactive": _summarize([row for row in rows if not row["active"]]),
    }
    for field, key in (("module", "by_module"), ("hook", "by_hook")):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        report[key] = {name: _summarize(group) for name, group in sorted(groups.items())}
    report["predictions"] = rows
    report["contract"] = (
        "Isolated call: return value, full memory, exception, and choose candidates. "
        "Global RNG state is recorded by the teacher but is not reproduced here; "
        "hidden module globals are outside this contract."
    )
    return report


def evaluate(
    model: Any,
    examples: Iterable[dict],
    *,
    mode: str = "hard",
    max_new_tokens: int | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Generate from each input alone and score the complete returned contract.

    ``output_text`` is used only to score a prediction; it is never passed to
    ``generate``.  The generation budget is independent of the
    expected output length.  With ``None``, the model uses its remaining context.
    """
    if mode not in {"hard", "soft"}:
        raise ValueError("mode must be hard or soft")
    rows = []
    for index, example in enumerate(examples):
        expected = decode_output(example["output_text"], example["hook"])
        row = _empty_row(example, index, expected)
        started = time.perf_counter()
        try:
            generated = model.generate(example["input_text"], max_new_tokens=max_new_tokens, mode=mode)
        except Exception as exc:
            row["status"] = "overflow" if type(exc).__name__ in {
                "ExampleTooLongError", "SequenceTooLongError", "ContextLengthError",
            } else "generation_error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["predicted_text"] = generated.get("text", "")
            row["stop_reason"] = generated.get("stop_reason")
            row["generated_tokens"] = generated.get("generated_tokens")
            row["input_tokens"] = generated.get("input_tokens")
            if generated.get("eos_generated") is not True:
                row["status"] = "eos_missing"
            elif generated.get("utf8_valid") is False:
                row["status"] = "invalid_utf8"
            else:
                try:
                    actual = decode_output(generated["text"], example["hook"])
                except json.JSONDecodeError as exc:
                    row["status"] = "invalid_json"
                    row["error"] = str(exc)
                except Exception as exc:
                    row["status"] = "invalid_schema"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                else:
                    row["status"] = "ok"
                    row.update(compare_outputs(expected, actual, example["hook"]))
        row["elapsed_seconds"] = time.perf_counter() - started
        rows.append(row)
        if progress is not None:
            progress(dict(row))
    return _report(rows, mode, "free_generation")


def always_none_baseline(examples: Iterable[dict]) -> dict:
    """A non-neural baseline which returns None and leaves all arguments alone."""
    rows = []
    for index, example in enumerate(examples):
        expected = decode_output(example["output_text"], example["hook"])
        row = _empty_row(example, index, expected)
        started = time.perf_counter()
        before = copy.deepcopy(_arguments(example))
        actual = {"return": None, "memory": before["memory"], "exception": None}
        if example["hook"] == "choose":
            actual["candidates"] = before["candidates"]
        row["status"] = "ok"
        row.update(compare_outputs(expected, actual, example["hook"]))
        row["elapsed_seconds"] = time.perf_counter() - started
        rows.append(row)
    return _report(rows, "none", "always_none_unchanged_arguments")


evaluate_baseline = always_none_baseline
