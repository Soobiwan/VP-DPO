"""Offline checks; these do not claim to execute a real language model."""

import copy
import csv
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import evaluation as ev


class InputTests(unittest.TestCase):
    def test_hub_card_and_local_checkpoint(self):
        expected = "meta-llama/Llama-3.2-1B"
        self.assertEqual(ev.normalize_model_source(expected), expected)
        self.assertEqual(ev.normalize_model_source(f"https://huggingface.co/{expected}/?x=1#model-card"), expected)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint with spaces"
            root.mkdir()
            (root / "config.json").write_text("{}")
            self.assertEqual(ev.normalize_model_source(root), str(root))
            with self.assertRaises(FileNotFoundError):
                ev.normalize_model_source(root / "missing")

    def test_dataset_urls_and_embedded_revisions_are_rejected(self):
        for source in ("https://huggingface.co/datasets/Idavidrein/gpqa",
                       "https://example.com/model/card",
                       "https://huggingface.co/org/model/tree/dev",
                       ""):
            with self.subTest(source=source), self.assertRaises(ValueError):
                ev.normalize_model_source(source)

    def test_limits_and_execution_settings(self):
        self.assertIsNone(ev.EvalConfig().limit)
        self.assertEqual(len(ev.EvalConfig().benchmarks), 6)
        for limit in (None, 2, 0.1):
            ev.EvalConfig(limit=limit).validate()
        for limit in (0, -1, True, 1.0, 2.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                ev.EvalConfig(limit=limit).validate()
        with self.assertRaisesRegex(ValueError, "HumanEval requires"):
            ev.EvalConfig(allow_humaneval_execution=False).validate()
        ev.EvalConfig(benchmarks=("MMLU",), allow_humaneval_execution=False).validate()
        with self.assertRaises(ValueError):
            ev.EvalConfig(benchmarks=("typo",)).validate()

    def test_auto_dtype_uses_fp16_on_turing_and_bf16_on_ampere(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device=lambda _: nullcontext(),
            get_device_capability=lambda _: (7, 5),
            # PyTorch may return True here even when BF16 is emulated.
            is_bf16_supported=lambda: True,
        )
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            self.assertEqual(ev.resolve_device_dtype("cuda:0", "auto"), ("cuda:0", "float16"))
            cuda.get_device_capability = lambda _: (8, 0)
            self.assertEqual(ev.resolve_device_dtype("cuda:0", "auto"), ("cuda:0", "bfloat16"))

    def test_notebook_is_clean_and_python_cells_compile(self):
        notebook = json.loads((HERE / "evaluate.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), f"cell-{index}", "exec")


def fake_result(tasks):
    return {
        "results": {task: {"acc,none": 0.5, "acc_stderr,none": 0.1} for task in tasks},
        "n-shot": {task: 0 for task in tasks},
        "n-samples": {task: {"original": 100, "effective": 2} for task in tasks},
        "samples": {task: [{"doc_id": 0, "resps": [["example"]]}] for task in tasks},
    }


class RunnerTests(unittest.TestCase):
    def fake_runtime(self, evaluator):
        stack = ExitStack()
        self.addCleanup(stack.close)
        package = types.ModuleType("lm_eval")
        package.simple_evaluate = evaluator
        tasks = types.ModuleType("lm_eval.tasks")
        tasks.TaskManager = Mock()
        stack.enter_context(patch.dict(sys.modules, {"lm_eval": package, "lm_eval.tasks": tasks}))
        stack.enter_context(patch.object(ev, "verify_harness_import"))
        stack.enter_context(patch.object(ev, "git_output", side_effect=lambda _, *args: "" if args[0] == "status" else ev.HARNESS_COMMIT))
        model = types.SimpleNamespace(device="cpu", model=types.SimpleNamespace(
            dtype="float32", config=types.SimpleNamespace(_commit_hash="model-sha")))
        loader = stack.enter_context(patch.object(ev, "load_model", return_value=model))
        stack.enter_context(patch.dict("os.environ", {}, clear=False))
        return loader

    def test_suite_passes_protocol_and_saves_all_benchmarks_once(self):
        evaluator = Mock(side_effect=lambda **kw: fake_result(kw["tasks"]))
        loader = self.fake_runtime(evaluator)
        config = ev.EvalConfig(limit=2, apply_chat_template=True)
        with tempfile.TemporaryDirectory() as directory:
            run = ev.run_suite(config, Path(directory), Path(directory))
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(evaluator.call_count, 6)
            for name, call in zip(config.benchmarks, evaluator.call_args_list):
                kwargs = call.kwargs
                self.assertEqual(kwargs["tasks"], list(ev.BENCHMARKS[name].tasks))
                self.assertEqual(kwargs["num_fewshot"], ev.BENCHMARKS[name].fewshot)
                self.assertEqual(kwargs["apply_chat_template"], name != "HumanEval")
                self.assertEqual(kwargs["confirm_run_unsafe_code"], name == "HumanEval")
                self.assertEqual(kwargs["limit"], 2)
                self.assertNotIn("gen_kwargs", kwargs)  # Preserve task-specific generation settings.
                artifact = json.loads((run / name.lower() / "results.json").read_text())
                self.assertNotIn("samples", artifact)
                self.assertTrue(list((run / name.lower()).glob("samples_*.jsonl")))
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(manifest["limited_run"])
            with (run / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 7)  # TruthfulQA contributes two tasks.
            self.assertTrue(all(row["metric"] == "acc" for row in rows))

    def test_failure_keeps_prior_scores_and_marks_pending_benchmarks(self):
        evaluator = Mock(side_effect=[fake_result(["mmlu"]), RuntimeError("dataset access denied")])
        self.fake_runtime(evaluator)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "dataset access denied"):
                ev.run_suite(ev.EvalConfig(), Path(directory), Path(directory))
            run = next(Path(directory).iterdir())
            self.assertTrue((run / "mmlu/results.json").is_file())
            self.assertTrue((run / "summary.csv").is_file())
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["benchmarks"]["MMLU"]["status"], "completed")
            self.assertEqual(manifest["benchmarks"]["GSM8K"]["status"], "failed")
            self.assertEqual(manifest["benchmarks"]["GPQA"]["status"], "pending")

    def test_full_and_smoke_outputs_are_separate(self):
        evaluator = Mock(side_effect=lambda **kw: fake_result(kw["tasks"]))
        self.fake_runtime(evaluator)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = ev.run_suite(ev.EvalConfig(benchmarks=("MMLU",), log_samples=False), root, root)
            smoke = ev.run_suite(ev.EvalConfig(benchmarks=("MMLU",), limit=2), root, root)
            self.assertNotEqual(full, smoke)
            self.assertIn("_full_", full.name)
            self.assertIn("_smoke_", smoke.name)
            self.assertEqual(list((full / "mmlu").glob("samples_*.jsonl")), [])

    def test_aggregate_and_filter_names_are_preserved(self):
        result = fake_result(["mmlu_subject"])
        result["groups"] = {"mmlu": {"acc,none": 0.73, "acc_stderr,none": 0.02}}
        result["results"]["gsm8k"] = {"exact_match,strict-match": 0.1,
                                        "exact_match,flexible-extract": 0.2}
        original = copy.deepcopy(result)
        rows = ev.metric_rows("MMLU", result, limited=False)
        self.assertEqual(result, original)
        aggregate = next(row for row in rows if row["task"] == "mmlu")
        self.assertEqual(aggregate["value"], 0.73)
        self.assertEqual(aggregate["stderr"], 0.02)
        self.assertEqual({row["filter"] for row in rows if row["task"] == "gsm8k"},
                         {"strict-match", "flexible-extract"})

    def test_torch_style_metadata_can_be_serialized(self):
        class DType:
            def __str__(self):
                return "torch.bfloat16"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            ev.write_json(path, {"config": {"model_dtype": DType()}})
            self.assertEqual(json.loads(path.read_text())["config"]["model_dtype"], "torch.bfloat16")


if __name__ == "__main__":
    unittest.main()
