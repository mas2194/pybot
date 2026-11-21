"""Build a result summary from measured reports; this does not run training."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def fraction(report, field):
    counts = report["counts"]
    return f"{counts[field]}/{counts['examples']}"


def main():
    training, fp_training = load("training.json"), load("training-fp-control.json")
    hard, soft = load("hard-validation.json"), load("soft-validation.json")
    fp = load("fp-control-validation.json")
    baseline = hard["always_none_baseline"]
    measured = [("LUT model / hard", hard), ("LUT model / soft", soft),
                ("FP control / eval", fp), ("Always None + unchanged memory", baseline)]
    table = "\n".join(
        f"| {name} | {fraction(report, 'whole_call_exact')} | "
        f"{fraction(report, 'return_exact')} | {fraction(report, 'memory_exact')} | "
        f"{fraction(report, 'ok')} |" for name, report in measured)
    lut, control = training["training_log"], fp_training["training_log"]
    summary = {
        "status": "complete-I/O reproduction not achieved",
        "teacher_revision": training["source_commits"],
        "griffith_revision": training["griffith_revision"],
        "train_records": training["split"]["train_records"],
        "validation_records": training["split"]["validation_records"],
        "coverage": {"teacher_modules": 102, "teacher_hooks": 116,
                     "natural_trace_records": 232, "independent_records": 120,
                     "trained_modules": 4, "trained_hooks": 6},
        "metrics": {name: {"counts": report["counts"], "rates": report["rates"]}
                    for name, report in measured},
        "scope": "One-call evaluation with recorded input memory; no learned full-chat rollout.",
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    text = f"""# Complete module I/O: measured experiment

**Full module I/O reproduction was not achieved.** The small learned LUT model
matched **0/18** held-out complete calls. The official Griffith floating-point
control also matched **0/18**. These weights document failed experimental baselines;
they are not working replacements for PyBot modules.

## Implemented and exercised scope

- Original PyBot revision: `{training['source_commits'][0]}`.
- Official Griffith revision: `{training['griffith_revision']}`.
- All 116 phase hooks in 102 modules were discovered. Two real conversation turns
  produced 232 complete records, covering all 116 hooks.
- The bounded learning experiment used 120 independent seeded calls in four
  modules/six hooks. The target is return + full updated memory + exception,
  together with updated candidate arguments for choose calls.
- 102 training and 18 validation records; zero truncated or length-excluded
  records. A colour, age pair, or name and its paraphrases stay in one group.
  Identical inputs cannot cross the split.
- Validation has 14 active and four inactive calls, in five hooks/three modules.
  The cake task has only one family, retained in training: it has no held-out
  coverage or generalization claim.

## Input-only free generation

Generation receives the complete input and no correct output prefix. The output
budget is the remaining model context, independent of the expected answer length.
Success requires valid typed JSON, the expected schema, EOS, and equality of all
output values, types, order, and shared references.

| Model | Whole call exact | Return exact | Memory exact | Valid complete output |
|---|---:|---:|---:|---:|
{table}

The always-None baseline succeeds on the four inactive calls and **0/14 active
calls**. Its 14/18 return-only score shows why judging only the return value would
hide missing memory updates.

All 18 hard and all 18 soft LUT predictions were invalid JSON. The FP control
produced 12 invalid-JSON predictions, one invalid-schema prediction, and five valid
completed outputs. None of those five had the correct full memory. None of the
three runs had context overflow or missing EOS.

## Training diagnostics are separate from reproduction accuracy

Both runs used identical training/validation fingerprints, width 64, one layer,
two heads, context 1280, batch four, 1,000 steps, learning rate 0.003, train seed 0,
and split seed 1. The LUT model uses native Griffith hard straight-through training.
The control uses official `attention='fp', ffn='fp'`. Parameter counts differ.

| Quantity | LUT model | FP control |
|---|---:|---:|
| Parameters | {lut['parameters']:,} | {control['parameters']:,} |
| Training seconds | {lut['training_seconds']:.3f} | {control['training_seconds']:.3f} |
| Training token accuracy, teacher-forced | {100*lut['teacher_forced_after']['hard']['teacher_forced_token_accuracy']:.4f}% | {100*control['teacher_forced_after']['hard']['teacher_forced_token_accuracy']:.4f}% |
| Training NLL, teacher-forced | {lut['teacher_forced_after']['hard']['output_nll_nats']:.6f} | {control['teacher_forced_after']['hard']['output_nll_nats']:.6f} |
| Training sequence accuracy, teacher-forced | {100*lut['teacher_forced_after']['hard']['teacher_forced_sequence_accuracy']:.4f}% | {100*control['teacher_forced_after']['hard']['teacher_forced_sequence_accuracy']:.4f}% |

CPU, one PyTorch thread, `{lut['torch_version']}`. Independent processes sometimes
ran concurrently, so latency is not a controlled speed comparison. `hard` in the
FP report is an evaluation-mode argument; the control has no LUTs to discretize.

## Interpretation

The implementation now trains and measures the requested object, a whole module
call including its state change. The initial model does not solve it. The FP
control reduces training loss substantially but still fails the held-out
full-state task. The experiment therefore does not isolate the failure to LUT
discretization or prove the module mapping cannot be learned.

For example, the FP control accepts `my favourite colour is purple` but stores
`orange` instead of `purple`, while returning None correctly. On the recall call,
it answers `silver` and also changes the stored colour to `silver`. Both are
validly formatted outputs and both fail whole-call equivalence. These are
predictions 0 and 2 in the FP validation report.

The result concerns this small byte-generating model, dataset, and training
budget. Less verbose lossless targets, generic copying of input values, and more
diverse training data are plausible next improvements; none is claimed here.

## Exact contract and limitations

The contract covers explicit hook arguments, their updated memory/candidate
values, return values, and exception descriptors. Teacher data executes original
parsing and knowledge lookup. Student generation imports no bot modules and uses
no teacher fallback, answer cache, or handcrafted response templates.

Independent fixtures initialize and restore the external RNG. Natural traces
record full pre/post RNG states and include the pre-call state in model inputs.
The model does not reproduce the process RNG side effect. Dates are recorded, not
virtualized; module-global caches are not captured. Cycles and custom objects
are rejected. This is not arbitrary-Python-process or unlimited-input equivalence.

Validation supplies the original input memory to each call. Predicted-memory
rollouts through a complete conversation were not evaluated. The collector and
contract support choose/after, but this pilot does not train those hooks. The
original `python -m pybot` runtime has not been switched to the failed models.

## Reproduce and inspect

The exact LUT commands are in the [repository README](../../README.md). For the
single FP control, add `--fp-baseline` to the same train command, use checkpoint
`fp-control.pt` and report `training-fp-control.json`, then evaluate its saved
validation split with `--mode hard`. Both completed runs were executed once,
without held-out tuning. An initial shape-configuration error stopped before
training; matching the official attention width to the residual width fixed it.

- [LUT training](training.json), [hard predictions](hard-validation.json),
  [soft predictions](soft-validation.json).
- [FP training](training-fp-control.json), [FP predictions](fp-control-validation.json).
- [Machine-readable summary](summary.json).
- Both weights and exact train/validation split records are in this directory.
- [Japanese interface and examples](../../docs/moduleio/README.ja.md).

A valid-looking generated value alone does not establish agreement with the
original module. The saved weights are experimental failure artifacts.
"""
    (ROOT / "RESULTS.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
