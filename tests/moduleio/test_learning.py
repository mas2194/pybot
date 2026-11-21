"""Learning contracts: full outputs, causal inputs, native LUTs and hard inference."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HAS_MODEL = (importlib.util.find_spec("torch") is not None and
             importlib.util.find_spec("griffith_logic") is not None)


@unittest.skipUnless(HAS_MODEL, "install the module-I/O extra (Griffith and torch)")
class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def model(self, length=96):
        import torch
        from moduleio.model import ModelConfig, ModuleIOModel
        torch.manual_seed(91)
        return ModuleIOModel(ModelConfig(max_seq_len=length, d_model=8,
                                        qk_bits=8, v_bits=8)).eval()

    def test_output_only_mask_covers_all_bytes_and_real_eos(self):
        from moduleio.model import EOS, IGNORE_INDEX, SEP, encode_text, prepare_example
        example = prepare_example('日本語', '{"return":null}', 96)
        self.assertEqual(example["input_ids"][len(example["prompt_ids"]) - 1], SEP)
        targets = [token for token in example["labels"] if token != IGNORE_INDEX]
        self.assertEqual(targets, encode_text('{"return":null}') + [EOS])
        self.assertEqual(len(example["input_ids"]), len(example["labels"]))
        self.assertEqual(example["answer_ids"], targets)
        empty = prepare_example("", "", 4)
        self.assertEqual(empty["labels"], [IGNORE_INDEX, EOS])

    def test_custom_model_width_resolves_attention_width_and_backpropagates(self):
        from torch.nn import functional as F
        from griffith_logic.layers.lut import LutLayer
        from moduleio.model import (IGNORE_INDEX, VOCAB_SIZE, ModelConfig,
                                    ModuleIOModel, prepare_example)
        from moduleio.training import collate
        config = ModelConfig(max_seq_len=32, d_model=64)
        self.assertEqual((config.qk_bits, config.v_bits), (64, 64))
        self.assertEqual(config, ModelConfig(max_seq_len=32, d_model=64,
                                            qk_bits=64, v_bits=64))
        with self.assertRaisesRegex(ValueError, "v_bits == d_model"):
            ModelConfig(d_model=64, v_bits=32)
        model = ModuleIOModel(config)
        batch = collate([prepare_example("in", "out", 32)])
        logits = model(batch["input_ids"], mode="soft")
        self.assertEqual(tuple(logits.shape), (*batch["input_ids"].shape, VOCAB_SIZE))
        mask = batch["labels"] != IGNORE_INDEX
        F.cross_entropy(logits[mask], batch["labels"][mask]).backward()
        layers = [part for part in model.modules() if isinstance(part, LutLayer)]
        self.assertTrue(any(part.weights.grad is not None and
                            part.weights.grad.abs().sum() > 0 for part in layers))

    def test_overlength_input_or_output_is_rejected_without_truncation(self):
        from moduleio.model import ExampleTooLongError, prepare_example
        for prompt, answer in (("x" * 100, "{}"), ("{}", "y" * 100)):
            with self.subTest(prompt=prompt), self.assertRaises(ExampleTooLongError):
                prepare_example(prompt, answer, 32)
        model = self.model(length=32)
        with self.assertRaises(ExampleTooLongError):
            model.generate("x" * 32, max_new_tokens=1)
        with self.assertRaises(ExampleTooLongError):
            model.generate("x", max_new_tokens=32)

    def test_labels_cannot_enter_prompt_and_future_output_is_causally_masked(self):
        import torch
        from moduleio.model import prepare_example
        model = self.model()
        a = prepare_example('{"x":1}', 'alpha', 96)
        b = prepare_example('{"x":1}', 'zebra', 96)
        self.assertEqual(a["prompt_ids"], b["prompt_ids"])
        boundary = len(a["prompt_ids"])
        for mode in ("soft", "hard"):
            one = model(torch.tensor([a["input_ids"]]), mode=mode)
            two = model(torch.tensor([b["input_ids"]]), mode=mode)
            self.assertTrue(torch.allclose(one[:, :boundary], two[:, :boundary], atol=1e-6))

    def test_native_griffith_parameters_receive_gradients_and_modes_restore(self):
        import torch
        from torch.nn import functional as F
        from griffith_logic.layers.lut import LutLayer
        from griffith_logic.models import LogicLadderGPT
        from moduleio.model import IGNORE_INDEX, prepare_example
        from moduleio.training import collate
        model = self.model()
        self.assertIsInstance(model.network, LogicLadderGPT)
        layers = [layer for layer in model.modules() if isinstance(layer, LutLayer)]
        self.assertTrue(layers)
        batch = collate([prepare_example("a", "b", 96), prepare_example("c", "d", 96)])
        settings = [(part.training, getattr(part, "gumbel", None),
                     getattr(part, "selection", None)) for part in model.modules()]
        mask = batch["labels"] != IGNORE_INDEX
        for mode in ("soft", "ste"):
            model.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], mode=mode)
            F.cross_entropy(logits[mask], batch["labels"][mask]).backward()
            self.assertTrue(any(layer.weights.grad is not None and
                                torch.count_nonzero(layer.weights.grad) for layer in layers))
            self.assertEqual(settings, [(part.training, getattr(part, "gumbel", None),
                getattr(part, "selection", None)) for part in model.modules()])

    def test_soft_and_hard_generation_are_deterministic_and_report_stopping(self):
        model = self.model()
        for mode in ("soft", "hard"):
            a = model.generate("{}", max_new_tokens=5, mode=mode)
            b = model.generate("{}", max_new_tokens=5, mode=mode)
            self.assertEqual(a, b)
            self.assertIn(a["stop_reason"], ("eos", "max_new_tokens"))
            self.assertLessEqual(a["generated_tokens"], 5)
        empty = model.generate("{}", max_new_tokens=0)
        self.assertEqual(empty["text"], "")
        self.assertFalse(empty["eos_generated"])
        self.assertEqual(empty["stop_reason"], "max_new_tokens")

    def test_hard_checkpoint_roundtrip_works_with_teacher_imports_blocked(self):
        from moduleio.model import load_checkpoint, save_checkpoint
        model = self.model()
        expected = model.generate("{}", max_new_tokens=4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            save_checkpoint(model, path, {"purpose": "contract test"})
            restored, metadata = load_checkpoint(path)
            self.assertEqual(restored.generate("{}", max_new_tokens=4), expected)
            self.assertEqual(metadata, {"purpose": "contract test"})
            script = '''
import importlib.abc
import json
import sys
class BlockTeacher(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bot" or fullname.startswith("bot."):
            raise AssertionError("teacher import attempted: " + fullname)
sys.meta_path.insert(0, BlockTeacher())
import torch
torch.set_num_threads(1)
from moduleio.model import load_checkpoint
model, _ = load_checkpoint(sys.argv[1])
print(json.dumps(model.generate("{}", max_new_tokens=4)))
assert not any(name == "bot" or name.startswith("bot.") for name in sys.modules)
'''
            result = subprocess.run([sys.executable, "-c", script, str(path)],
                                    cwd=Path(__file__).resolve().parents[2],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), expected)

    def test_tiny_fit_uses_all_output_tokens_and_reports_teacher_forcing(self):
        from moduleio.model import ModelConfig
        from moduleio.training import fit, teacher_forced_metrics
        examples = [{"input_text": "x", "output_text": "{}"}]
        config = ModelConfig(max_seq_len=24, d_model=8, qk_bits=8, v_bits=8)
        model, report = fit(examples, config, steps=3, batch_size=1)
        self.assertEqual(report["truncated_examples"], 0)
        self.assertEqual(report["max_output_tokens_including_eos"], 3)
        metrics = teacher_forced_metrics(model, examples)
        self.assertEqual(metrics["output_tokens_including_eos"], 3)
        self.assertIn("not free generation", metrics["conditioning"])


if __name__ == "__main__":
    unittest.main()
