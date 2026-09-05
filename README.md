# VP-DPO / BeeS experiments

This repository contains preference-data selection, DPO-family training, ranking, and evaluation experiments built around BeeS, Llama 3.2, OLMo 2 1B, and MiniCPM. The repository is organized by purpose so that runnable notebooks, reusable code, input data, configuration, and generated output do not get mixed together.

## Repository map

| Path | Contents |
| --- | --- |
| `BeeS/` | Main Python implementation, training/evaluation modules, notebook builders, prompt templates, and the canonical two-notebook OLMo workflow. |
| `BeeS/notebooks/` | Start-to-finish local OLMo 2 workflow: prepare/select data, then train and evaluate DPO. |
| `notebooks/local/` | Local workstation experiments, including Ollama ranking and the multi-method dual-GPU run. |
| `notebooks/kaggle/` | Self-contained Kaggle notebooks for MiniCPM, OLMo, and Llama methods on P100 or dual T4 GPUs. |
| `notebooks/kaggle/llama32_1b_training/` | One-method-per-notebook Llama 3.2 1B training suite for Kaggle T4 x2; all non-TIDPO methods use the canonical BeeS JSONL. |
| `notebooks/kaggle/llama32_3b_training/` | Separate Llama 3.2 3B version of the same non-TIDPO Kaggle T4 x2 training suite. |
| `notebooks/kaggle/per_model_dual_t4/` | One-model-per-notebook, dual-T4 evaluation suite for MMLU, GSM8K, GPQA, HumanEval, TruthfulQA, and IFEval. |
| `notebooks/evaluation/` | AlpacaEval judge notebook and the self-contained [Hugging Face harness suite](notebooks/evaluation/lm_harness/README.md) for six benchmarks. |
| `scripts/evaluation/` | Command-line AlpacaEval runner and Azure OpenAI compatibility adapter. |
| `scripts/ranking/` | Resumable Ollama segmenter/ranker used by the local ranking notebook. |
| `data/processed/` | Tracked BeeS-selected and segmented UltraFeedback JSONL data plus its manifest. |
| `configs/ollama/` | Ollama model definition used by the ranking workflow. |
| `third_party/TIDPO/` | Imported TIDPO implementation, tests, configuration, and upstream notes. |
| `artifacts/` | Generated datasets, checkpoints, evaluations, and approvals. Ignored by Git. |
| `.cache/`, `.tmp/` | Download caches and scratch files. Ignored by Git. |

## Recommended workflow

The clearest reproducible path is the OLMo 2 1B BeeS workflow:

1. Run `BeeS/notebooks/01_create_bees_ultrafeedback_olmo2.ipynb` to download UltraFeedback, prepare preference pairs, train the proxy scorer, and select the BeeS subset.
2. Run `BeeS/notebooks/02_train_olmo2_1b_dpo.ipynb` to train full-parameter DPO and execute the quality gate.
3. Optionally run `notebooks/evaluation/alpaca-eval-2-judge.ipynb` directly, or use the command-line wrapper described below.

See `BeeS/OLMO2_BEES_README.md` for the design, GPU-memory strategy, checkpoint behavior, pinned model/dataset revisions, and accuracy gate.

For the current Llama experiments, choose one notebook from
`notebooks/kaggle/llama32_1b_training/` or `notebooks/kaggle/llama32_3b_training/`.
The suites cover Simple DPO, VPDPO A, B/B_norm/C with DPO or VDPO cores, SimPO, and SamPO.
TIDPO is intentionally excluded until its original notebook is corrected.

## Setup

Python 3.10 or 3.11 and an NVIDIA CUDA environment are recommended. The OLMo notebooks expect two CUDA GPUs; the Kaggle filenames identify their intended accelerator.

From the repository root:

```bash
python -m venv .venv
```

Activate it on PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install a CUDA-compatible PyTorch build for your machine first, then install the pinned workflow dependencies:

```bash
python -m pip install -r BeeS/requirements-olmo2.txt
python -m ipykernel install --user --name vpdpo-olmo --display-name "VP-DPO (OLMo)"
```

Copy `.env.example` to `.env`. Hugging Face credentials are optional for public downloads; Azure credentials are required only for the Azure-backed AlpacaEval judge.

```powershell
Copy-Item .env.example .env
```

Never commit `.env`; it is already ignored.

## Running notebooks

Launch Jupyter from the repository root so all workspace-relative paths resolve consistently:

```bash
jupyter lab
```

Choose the `VP-DPO (OLMo)` kernel, then open the relevant notebook:

- `BeeS/notebooks/`: primary local preparation and DPO workflow.
- `notebooks/local/`: local Ollama ranking and experimental multi-method training.
- `notebooks/kaggle/`: upload the selected notebook to Kaggle and use the accelerator named in its filename. These notebooks manage Kaggle input/output paths themselves.
- `notebooks/kaggle/llama32_1b_training/` and `notebooks/kaggle/llama32_3b_training/`: accept the Llama 3.2 license, add a `Huggingface` or `HF_TOKEN` Kaggle secret, select T4 x2, and run one method notebook per session.
- `notebooks/kaggle/per_model_dual_t4/`: choose one model notebook, attach its configured Kaggle input, select GPU T4 x2, and run all cells to produce raw artifacts, scores, and a ZIP before moving to the next model.
- `notebooks/evaluation/`: interactive AlpacaEval judging.

The notebooks expose their run switches and paths near the top. Read those configuration cells before running all cells because the training jobs are long and produce large checkpoints.

## AlpacaEval from the command line

After a model has been approved by the OLMo quality gate and `.env` contains the Azure settings:

```bash
python scripts/evaluation/run_alpaca_eval2.py
```

Useful safe checks:

```bash
python scripts/evaluation/run_alpaca_eval2.py --help
python scripts/evaluation/run_alpaca_eval2.py --limit 2 --generation-only
```

By default the runner reads the approved model record under `artifacts/olmo2_bees/`, writes generations under `artifacts/olmo2_bees/alpaca_eval2/`, and executes `notebooks/evaluation/alpaca-eval-2-judge.ipynb`. Use `--model`, `--output`, or `--notebook` to override those locations.

## Ollama ranking

Create the configured Ollama model from the repository root:

```bash
ollama create qwen3.8:27b-q3 -f configs/ollama/Modelfile.qwen3.8-27b-q3
```

Start Ollama, then open `notebooks/local/ollama_ranking_olmo_bees.ipynb`. The reusable and resumable implementation is in `scripts/ranking/ollama_olmo_bees_ranker.py`; the canonical ranking prompts remain in `notebooks/kaggle/kaggle_ranking.ipynb`.

## Regenerating maintained notebooks

Run these from the repository root after changing notebook-builder code:

```bash
python BeeS/tools/build_notebooks.py
python BeeS/tools/build_structured_notebook.py
python BeeS/tools/build_kaggle_preference_notebooks.py
python BeeS/tools/build_kaggle_method_b_norm_notebooks.py
python BeeS/tools/build_kaggle_llama32_training_notebooks.py
python scripts/build_dual_t4_eval_notebooks.py
```

The builders write into `BeeS/notebooks/`, `notebooks/local/`, or `notebooks/kaggle/` as appropriate.

## Data and output policy

The three files in `data/processed/` are intentional, versioned experiment inputs. Do not put new model weights or generated runs there. Store generated datasets, reference caches, checkpoints, and evaluation output in `artifacts/`; transient downloads and scratch data belong in `.cache/` and `.tmp/`. These output locations are excluded by `.gitignore`.

## Legacy BeeS scripts

The original scripts directly under `BeeS/` retain upstream assumptions such as an `adpo` package/directory name and external `models/` or `datasets/` folders. They are preserved for research provenance. For a clean first run, use the OLMo notebooks and `BeeS/olmo2_bees/` modules documented above rather than the legacy shell scripts.

## Notes

- Full model training is GPU-intensive and can consume tens of GiB of disk per checkpoint. Check free space before starting.
- The quality gate is fail-closed: a completed training run is not automatically an approved model.
- `third_party/TIDPO/` has its own README, quickstart, license, and troubleshooting notes.
