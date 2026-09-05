# Hugging Face model evaluation notebook

Open [evaluate.ipynb](evaluate.ipynb) and run its cells from top to bottom. The
default is the **base** `meta-llama/Llama-3.2-1B` model and all six benchmarks.
The notebook installs the chosen EleutherAI harness checkout, checks access,
loads the model once, evaluates each benchmark, and displays/saves scores.

## Start

Use Python 3.10–3.12, Git, and preferably a Linux NVIDIA GPU runtime. From this
folder, create a separate environment and launch Jupyter:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab ipykernel
python -m jupyter lab
```

Install a PyTorch build appropriate for your runtime if necessary; the notebook
installs its remaining dependencies in the active kernel. The default batch size
is 1 and context limit is 4,096 tokens. CUDA automatically uses native BF16 on
Ampere-or-newer GPUs and FP16 on older GPUs such as Turing; CPU uses FP32 and can
take much longer. Use `DEVICE="cuda:0"`
to require CUDA instead of allowing the automatic CPU fallback. This is a single
process workflow and does not require two GPUs.

Before using the default inputs, obtain access to the
[Llama model](https://huggingface.co/meta-llama/Llama-3.2-1B) and
[GPQA dataset](https://huggingface.co/datasets/Idavidrein/gpqa). Set `HF_TOKEN` in
your environment or enter it in the notebook's hidden prompt. A token with read
access is sufficient once your account has access to both resources.

**HumanEval executes model-generated Python on the runtime.** Run the full suite
in a disposable, isolated runtime. The notebook enables both code-execution
settings required by the harness. Set `ALLOW_HUMANEVAL_EXECUTION=False` and remove
`"HumanEval"` from `SELECTED_BENCHMARKS` if code execution is unsuitable. The
upstream metric's subprocess/time limits are not a security sandbox. Use Linux
or WSL for this benchmark.

## Inputs

- `MODEL_SOURCE`: Hub ID, root model-card URL, or a local HF checkpoint directory.
  Examples: `meta-llama/Llama-3.2-1B`,
  `https://huggingface.co/meta-llama/Llama-3.2-1B`, `/path/to/checkpoint`.
  A model card identifies the weights; Markdown alone is not a runnable model.
- `MODEL_REVISION`: Hub branch, tag, or commit. Preflight resolves this to a commit
  before loading a remote model. Put revisions here rather than in the card URL.
- `HARNESS_SOURCE`: Git repository URL or an existing local harness checkout.
  The default clones release `v0.4.11` into `.cache/lm-evaluation-harness` and
  verifies commit `27988a293647d5853e48edea291640b4af54740c`. For an existing
  checkout at another commit, set `HARNESS_REF=None`. Existing checkouts are used
  unchanged; the notebook checks the imported package really comes from that path.
  Other harness versions must support the same API/task names.
- `SELECTED_BENCHMARKS`: any subset of the six names below.
- `SMOKE_TEST`: default `False` runs the full suite. `True` sets a limit of two
  examples **per task**, including each of the 57 MMLU subjects. Such scores only
  validate plumbing and must not be presented as full benchmark scores.

For a PyTorch model configured for Transformers, save both model and tokenizer:

```python
model.save_pretrained("/path/to/checkpoint", safe_serialization=True)
tokenizer.save_pretrained("/path/to/checkpoint")
# Then use that directory as MODEL_SOURCE.
```

Alternatively, assign your `transformers.PreTrainedModel` and matching tokenizer
to `LOADED_MODEL` and `LOADED_TOKENIZER` in the notebook. Place the loaded model on
your desired device/dtype first; HFLM keeps its existing placement. Its weights
take precedence over `MODEL_SOURCE`. Plain state dictionaries and arbitrary
`torch.nn.Module` objects require a Transformers-compatible wrapper/export first.
An adapter-only directory should be loaded/merged with its base model before
exporting a complete checkpoint.

For an instruction-tuned checkpoint, set `APPLY_CHAT_TEMPLATE=True` with its
matching tokenizer. The base Llama default uses raw prompts. HumanEval always
uses the stock raw code-completion prompt, including when other tasks use chat
templates; evaluating instruction-style code output requires a different task.

## Evaluation protocol

| Benchmark | Harness task(s) | Few-shot | Reported metrics |
| --- | --- | ---: | --- |
| MMLU | `mmlu` (all 57 subjects) | 5 | Harness aggregate and subject accuracy |
| GSM8K | `gsm8k` | 5 | Exact match, strict and flexible answer extraction |
| GPQA | `gpqa_main_zeroshot` | 0 | Main split accuracy and normalized accuracy |
| HumanEval | `humaneval` | 0 | Greedy single-completion pass@1 |
| TruthfulQA | `truthfulqa_mc1`, `truthfulqa_mc2` | 0* | MC1 accuracy and MC2 probability mass |
| IFEval | `ifeval` | 0 | Prompt/instruction accuracy, strict and loose |

\* TruthfulQA uses the fixed examples embedded in the harness prompt, with zero
additional sampled few-shot examples. TruthfulQA here covers multiple choice;
it does not run the separate free-form generation/judge task. GPQA uses **Main**,
not Diamond or all overlapping GPQA variants. IFEval and GPQA use their datasets'
`train`-named splits for evaluation, as specified by the harness.

Task prompts, extraction, stop sequences, generation budgets, and scoring come
from the selected harness release. MMLU's aggregate comes from the harness rather
than an unweighted average of subject scores. Seeds are 42 and bootstrap
iterations are 1,000. There is no single score averaged across benchmarks. The
4,096-token context cap may truncate long prompts; increase `MAX_LENGTH` with
adequate memory when comparing against another evaluation protocol. These
settings do not claim to reproduce Meta's published model-card numbers.

## Files and results

- `evaluate.ipynb`: configuration, installation, authentication, preflight,
  execution, and result display.
- `evaluation.py`: reusable configuration, setup, model loading, and suite runner.
- `requirements.txt`: dependency compatibility bounds; the harness revision is
  pinned and actual installed package versions are recorded per run.
- `tests/test_evaluation.py`: offline configuration and orchestration checks.
- `.cache/`: downloaded harness and caches, ignored by Git.
- `artifacts/<UTC timestamp>_<full or smoke>_<id>/`: separate directory per run.

Every run saves `manifest.json` (settings, actual harness commit, dirty status,
model revision/device/dtype, package versions, and completion status),
`summary.csv`, and `<benchmark>/results.json`. With `LOG_SAMPLES=True`, sample
inputs, model outputs, and scoring details are saved as JSONL per harness task.
These sample files may include dataset questions; keep them with your local
evaluation artifacts. A loaded model's in-memory weight edits cannot be identified
by its base Hub revision; export a checkpoint for reproducible comparisons.

Failures raise an exception and mark the manifest as failed; completed results
remain on disk. After resolving access, memory, or dependency issues, rerun just
the failed/pending benchmarks by editing `SELECTED_BENCHMARKS`. The new run gets
its own folder; there is no automatic reuse of potentially stale scores. After
installing or changing the harness in an already-used kernel, restart the kernel
and rerun with `INSTALL_DEPENDENCIES=False`.

Offline checks (no GPU/downloads):

```bash
python3 -m unittest discover -s notebooks/evaluation/lm_harness/tests -v
```

Run that command from the repository root. A real smoke/full evaluation requires
the notebook dependencies, model/dataset access, and a suitable runtime.

## Upstream references

- [Harness installation and model backends](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/README.md)
- [HFLM model wrapper](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/models/huggingface.py)
- [Evaluation API](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/evaluator.py)
- [Task definitions](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11/lm_eval/tasks)
