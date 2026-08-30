"""Build one resumable dual-T4 evaluation notebook per configured model.

The large, multi-model notebook remains untouched.  This builder uses its current
source cells as the evaluation/scoring implementation, removes historical outputs,
and replaces the model loop with a two-replica benchmark scheduler.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "notebooks" / "kaggle" / "tidpo_generation_suite.ipynb"
OUTPUT_DIR = REPO_ROOT / "notebooks" / "kaggle" / "per_model_dual_t4"

MODELS = [
    {
        "number": 1,
        "name": "VPDPO_B_Norm_DPO",
        "filename": "01_vpdpo_b_norm_dpo_dual_t4.ipynb",
        "path": "/kaggle/input/datasets/lonnyng/b-norm-dpo-olmo-1b/olmo2_bees_b_norm_dpo/run/final",
    },
    {
        "number": 2,
        "name": "VPDPO_B_Norm_VDPO",
        "filename": "02_vpdpo_b_norm_vdpo_dual_t4.ipynb",
        "path": "/kaggle/input/datasets/ahmadsubhaniiqbal/b-norm-vpdpo-olmo-1b/olmo2_bees_b_norm_vdpo/run/final",
    },
    {
        "number": 3,
        "name": "VPDPO_B_DPO",
        "filename": "03_vpdpo_b_dpo_dual_t4.ipynb",
        "path": "/kaggle/input/datasets/ahmadsubhaniiqbal/olmo-1b-b-d",
    },
    {
        "number": 4,
        "name": "VPDPO_B_VDPO",
        "filename": "04_vpdpo_b_vdpo_dual_t4.ipynb",
        "path": "/kaggle/input/datasets/ahmadsubhaniiqbal/olmo-1b-b-vdpo",
    },
    {
        "number": 5,
        "name": "VPDPO_C_DPO",
        "filename": "05_vpdpo_c_dpo_dual_t4.ipynb",
        "path": "/kaggle/input/datasets/ahmadsubhaniiqbal/olmo-1b-c-dpo",
    },
    {
        "number": 6,
        "name": "VPDPO_C_VDPO",
        "filename": "06_vpdpo_c_vdpo_dual_t4.ipynb",
        "path": "/kaggle/input/datasets/ahmadsubhaniiqbal/olmo-1b-c-vdpo",
    },
    {
        "number": 7,
        "name": "Simple_DPO",
        "filename": "07_simple_dpo_dual_t4.ipynb",
        "path": "/kaggle/input/datasets/ahmadsubhaniiqbal/olmo-1b-dpo-plain",
    },
    {
        "number": 8,
        "name": "VPDPO_A",
        "filename": "08_vpdpo_a_dual_t4.ipynb",
        "path": "/kaggle/input/datasets/ahmadsubhaniiqbal/olmo-1b-method-a",
    },
    {
        "number": 9,
        "name": "SimPO",
        "filename": "09_simpo_dual_t4.ipynb",
        "path": "/kaggle/input/datasets/jonathonbreaux/olmo2-bees-simpo/olmo2_bees_simpo/run/final",
    },
    {
        "number": 10,
        "name": "SAMPO",
        "filename": "10_sampo_dual_t4.ipynb",
        "path": "/kaggle/input/datasets/lonnyng/sampo-models/olmo2_bees_sampo/run/final",
    },
    {
        "number": 11,
        "name": "TIDPO",
        "filename": "11_tidpo_dual_t4.ipynb",
        "path": "/kaggle/input/datasets/ahmadsubhaniiqbal/olmo-bees-tidpo/olmo2_bees_tidpo/run/final",
    },
]


def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def find_cell(notebook: dict, marker: str, *, cell_type: str = "code") -> dict:
    matches = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == cell_type and marker in source_text(cell)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {cell_type} cell containing {marker!r}; found {len(matches)}")
    return matches[0]


def clean_notebook(notebook: dict) -> None:
    for cell in notebook["cells"]:
        cell["id"] = hashlib.sha256(source_text(cell).encode("utf-8")).hexdigest()[:8]
        cell.get("metadata", {}).pop("execution", None)
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def configuration_source(model: dict) -> str:
    output_slug = model["filename"].removesuffix("_dual_t4.ipynb").split("_", 1)[1]
    return f'''# This notebook intentionally evaluates exactly one model.
MODEL = {{
    "name": {model["name"]!r},
    "path": {model["path"]!r},
}}
MODELS = [MODEL]  # Kept as a one-item list for the shared validation/scoring helpers.

OUTPUT_ROOT = "/kaggle/working/tidpo_eval_{output_slug}"
HF_TOKEN_SECRET_NAME = "Huggingface"
SUITE_NAME = "tidpo_compatible_v1"

APPLY_CHAT_TEMPLATE = True
SEED = 42

# Use one complete model replica per T4 and dynamically schedule benchmarks between them.
MAX_GPU_WORKERS = 2
REQUIRE_DUAL_GPU = True
BENCHMARK_SCHEDULE = ["mmlu", "ifeval", "gsm8k", "truthfulqa", "gpqa", "humaneval"]

# End-to-end local scoring. HumanEval requires executing untrusted generated Python.
RUN_SCORING = True
EXECUTE_HUMANEVAL = True
HUMANEVAL_NUM_WORKERS = 4
HUMANEVAL_TIMEOUT_SECONDS = 3.0

# Fixed per-benchmark sizes avoid lm-eval's auto-batch divide-by-zero on small tail groups.
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_BATCH_SIZE = 16
BENCHMARK_BATCH_SIZES = {{
    "mmlu": 8,
    "gsm8k": 4,
    "gpqa": 8,
    "humaneval": 2,
    "truthfulqa": 8,
    "ifeval": 4,
}}
# Generation chunks are deliberately small: progress/checkpoints update after each chunk.
BENCHMARK_CHUNK_SIZES = {{
    "mmlu": 16,
    "gsm8k": 4,
    "gpqa": 16,
    "humaneval": 2,
    "truthfulqa": 16,
    "ifeval": 4,
}}
FLUSH_EVERY = 8
'''


OOM_HELPERS = r'''def clear_gpu_cache(gpu_index):
    gc.collect()
    with torch.cuda.device(int(gpu_index)):
        torch.cuda.empty_cache()


def inference_with_oom_backoff(model_context, method_name, requests):
    """Retry an inference call at progressively smaller batches after a CUDA OOM."""
    lm = model_context["lm"]
    while True:
        try:
            method = getattr(lm, method_name)
            return method(requests, disable_tqdm=True)
        except torch.OutOfMemoryError as exc:
            current_batch = int(lm.batch_size_per_gpu)
            if current_batch <= 1:
                raise
            error_summary = str(exc).splitlines()[0]
            # Drop traceback-held logits before asking the CUDA allocator to release them.
            exc.__traceback__ = None
            next_batch = max(1, current_batch // 2)
            lm.batch_size_per_gpu = next_batch
            lm.batch_schedule = 1
            lm.batch_sizes.clear()
            model_context.setdefault("batch_backoffs", {})[
                model_context.get("benchmark", "unknown")
            ] = next_batch
            clear_gpu_cache(model_context["gpu_index"])
            print(
                f"CUDA OOM on GPU {model_context['gpu_index']} during {method_name}; "
                f"retrying batch {current_batch} -> {next_batch}. {error_summary}"
            )
'''


MAIN_LOOP = r'''validate_model_configuration()
if len(MODELS) != 1:
    raise RuntimeError(f"This notebook must contain exactly one model; found {len(MODELS)}.")

configured_model = MODELS[0]
model_name = configured_model["name"]
model_slug = safe_slug(model_name)
model_dir = OUTPUT_ROOT_PATH / model_slug
model_dir.mkdir(parents=True, exist_ok=True)
resolved_model_path = resolve_model_path(configured_model["path"])
pre_context = {
    "model_name": model_name,
    "model_path": str(resolved_model_path),
    "model_dir": str(model_dir),
}
if model_name not in MANIFEST["models"]:
    MANIFEST["models"].append(model_name)
    atomic_write_json(MANIFEST_PATH, MANIFEST)

completion_state = {
    benchmark: benchmark_is_complete(pre_context, benchmark)
    for benchmark in BENCHMARK_ORDER
}
if all(done for done, _info in completion_state.values()):
    print(f"SKIP MODEL {model_name}: every benchmark passed DONE integrity checks.")
    for benchmark in BENCHMARK_ORDER:
        paths = benchmark_paths(model_dir, benchmark)
        records = repair_and_read_jsonl(paths["samples"])
        update_manifest(
            model_name,
            benchmark,
            "complete",
            len(records),
            sha256_file(paths["samples"]),
            paths["samples"],
        )
else:
    replicas = []
    pending_benchmarks = []
    try:
        for gpu_index in range(GPU_WORKER_COUNT):
            device = f"cuda:{gpu_index}"
            lm, replica_path, chat_metadata = load_model(configured_model, device=device)
            replicas.append(
                {
                    "lm": lm,
                    "model_name": model_name,
                    "model_path": str(replica_path),
                    "model_dir": str(model_dir),
                    "chat_template": chat_metadata,
                    "device": device,
                    "gpu_index": gpu_index,
                }
            )

        replica_paths = {context["model_path"] for context in replicas}
        chat_hashes = {context["chat_template"]["sha256"] for context in replicas}
        tokenizer_hashes = {
            canonical_json(context["chat_template"].get("tokenizer_loading"))
            for context in replicas
        }
        if len(replica_paths) != 1 or len(chat_hashes) != 1 or len(tokenizer_hashes) != 1:
            raise RuntimeError("The two GPU replicas did not resolve to identical model/tokenizer settings.")

        metadata = model_metadata(configured_model, replicas[0])
        metadata["inference_strategy"] = "one_full_model_replica_per_gpu_dynamic_benchmark_queue"
        metadata["replica_devices"] = [context["device"] for context in replicas]
        metadata["gpu_worker_count"] = len(replicas)
        metadata["benchmark_schedule"] = BENCHMARK_SCHEDULE
        metadata["benchmark_batch_sizes"] = BENCHMARK_BATCH_SIZES
        metadata["benchmark_chunk_sizes"] = BENCHMARK_CHUNK_SIZES
        write_or_validate_model_metadata(model_dir / "metadata.json", metadata)

        for benchmark in BENCHMARK_ORDER:
            done, _info = completion_state[benchmark]
            paths = benchmark_paths(model_dir, benchmark)
            if done:
                records = repair_and_read_jsonl(paths["samples"])
                update_manifest(
                    model_name,
                    benchmark,
                    "complete",
                    len(records),
                    sha256_file(paths["samples"]),
                    paths["samples"],
                )
            else:
                pending_benchmarks.append(benchmark)

        scheduled = [name for name in BENCHMARK_SCHEDULE if name in pending_benchmarks]
        if set(scheduled) != set(pending_benchmarks) or len(scheduled) != len(pending_benchmarks):
            raise RuntimeError("BENCHMARK_SCHEDULE must contain every benchmark exactly once.")

        job_queue = queue.Queue()
        outcome_queue = queue.Queue()
        for benchmark in scheduled:
            job_queue.put(benchmark)

        def gpu_lane(context):
            with torch.cuda.device(context["gpu_index"]):
                while True:
                    try:
                        benchmark = job_queue.get_nowait()
                    except queue.Empty:
                        return
                    try:
                        print(f"GPU {context['gpu_index']} claimed {benchmark}.")
                        result = run_benchmark(context, benchmark)
                        outcome_queue.put(
                            {"benchmark": benchmark, "status": "complete", "result": result}
                        )
                    except Exception as exc:
                        error_type = type(exc).__name__
                        error_message = str(exc)
                        formatted_traceback = traceback.format_exc()
                        # The exception traceback can retain large CUDA tensors after an OOM.
                        exc.__traceback__ = None
                        outcome_queue.put(
                            {
                                "benchmark": benchmark,
                                "status": "incomplete",
                                "error_type": error_type,
                                "error": error_message,
                                "traceback": formatted_traceback,
                            }
                        )
                        print(
                            f"ERROR GPU {context['gpu_index']} / {model_name} / {benchmark}: "
                            f"{error_type}: {error_message}"
                        )
                        print(formatted_traceback)
                        clear_gpu_cache(context["gpu_index"])
                    finally:
                        job_queue.task_done()

        print(f"Starting {len(replicas)} GPU lanes for {len(scheduled)} pending benchmarks.")
        with ThreadPoolExecutor(max_workers=len(replicas), thread_name_prefix="eval-gpu") as executor:
            futures = [executor.submit(gpu_lane, context) for context in replicas]
            for future in futures:
                future.result()

        outcomes = {}
        while not outcome_queue.empty():
            outcome = outcome_queue.get_nowait()
            outcomes[outcome["benchmark"]] = outcome

        for benchmark in pending_benchmarks:
            outcome = outcomes.get(benchmark)
            paths = benchmark_paths(model_dir, benchmark)
            records = repair_and_read_jsonl(paths["samples"])
            if outcome and outcome["status"] == "complete":
                result = outcome["result"]
                update_manifest(
                    model_name,
                    benchmark,
                    "complete",
                    result["samples"],
                    result["sha256"],
                    paths["samples"],
                )
                continue

            error_type = (outcome or {}).get("error_type", "WorkerError")
            error_message = (outcome or {}).get("error", "GPU worker returned no outcome.")
            paths["dir"].mkdir(parents=True, exist_ok=True)
            error_payload = {
                "status": "incomplete",
                "error_type": error_type,
                "error": error_message,
                "samples": len(records),
                "expected_samples": len(EXPECTED_IDS[benchmark]),
                "updated_at": utc_now(),
            }
            atomic_write_json(paths["incomplete"], error_payload)
            update_manifest(
                model_name,
                benchmark,
                "incomplete",
                len(records),
                sha256_file(paths["samples"]) if paths["samples"].exists() else None,
                paths["samples"],
                error=f"{error_type}: {error_message}",
            )
    except Exception as exc:
        print(f"MODEL SETUP ERROR {model_name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        for benchmark in BENCHMARK_ORDER:
            paths = benchmark_paths(model_dir, benchmark)
            done, _info = benchmark_is_complete(pre_context, benchmark)
            if done:
                continue
            paths["dir"].mkdir(parents=True, exist_ok=True)
            records = repair_and_read_jsonl(paths["samples"])
            error_payload = {
                "status": "incomplete",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "samples": len(records),
                "expected_samples": len(EXPECTED_IDS[benchmark]),
                "updated_at": utc_now(),
            }
            atomic_write_json(paths["incomplete"], error_payload)
            update_manifest(
                model_name,
                benchmark,
                "incomplete",
                len(records),
                sha256_file(paths["samples"]) if paths["samples"].exists() else None,
                paths["samples"],
                error=f"{type(exc).__name__}: {exc}",
            )
    finally:
        for context in replicas:
            lm = context.get("lm")
            if lm is None:
                continue
            try:
                del lm._model
            except Exception:
                pass
            context["lm"] = None
        replicas.clear()
        gc.collect()
        for gpu_index in range(torch.cuda.device_count()):
            with torch.cuda.device(gpu_index):
                torch.cuda.empty_cache()
        print(f"Released all model replicas for {model_name}.")
'''


def build_notebook(template: dict, model: dict) -> dict:
    notebook = copy.deepcopy(template)
    clean_notebook(notebook)

    intro = notebook["cells"][0]
    intro["source"] = source_lines(
        f'''# {model["name"]}: dual-T4 end-to-end evaluation

This Kaggle notebook evaluates **only `{model["name"]}`** over MMLU, GSM8K, full GPQA Main,
HumanEval, TruthfulQA MC2, and IFEval, then computes its local scores and exports a ZIP.

It loads one complete model replica on each T4 and dynamically gives each GPU one benchmark
at a time. Raw artifacts are append-only and each benchmark has an integrity-checked `DONE`
marker, so rerunning the main cell resumes unfinished work. HumanEval executes untrusted
generated Python in timed worker processes; a raw-artifact ZIP is created first.

Attach the configured Kaggle model input, enable **GPU T4 x2**, enable Internet, add the
`Huggingface` secret (with GPQA access), and run all cells.
'''
    )

    config_cell = find_cell(notebook, "MODELS = [")
    config_cell["source"] = source_lines(configuration_source(model))

    dependency_cell = find_cell(notebook, "PINNED_PACKAGES = [")
    dependency_source = source_text(dependency_cell)
    dependency_source = dependency_source.replace('    "numexpr==2.10.2",\n', "")
    dependency_cell["source"] = source_lines(dependency_source)

    diagnostics_cell = find_cell(notebook, "OUTPUT_ROOT_PATH = Path(OUTPUT_ROOT)")
    diagnostics_source = source_text(diagnostics_cell)
    diagnostics_source = diagnostics_source.replace(
        "import zipfile\n",
        "import zipfile\nimport queue\nimport threading\nfrom concurrent.futures import ThreadPoolExecutor\n",
    )
    diagnostics_source = diagnostics_source.replace(
        'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")\n',
        'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")\n'
        'os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")\n',
    )
    gpu_guard = '''if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. In Kaggle, select Settings > Accelerator > GPU T4 x2.")
'''
    replacement_guard = gpu_guard + '''if REQUIRE_DUAL_GPU and torch.cuda.device_count() < 2:
    raise RuntimeError(
        f"This notebook requires two visible GPUs, but found {torch.cuda.device_count()}. "
        "In Kaggle, select the GPU T4 x2 accelerator."
    )
GPU_WORKER_COUNT = min(int(MAX_GPU_WORKERS), torch.cuda.device_count())
if GPU_WORKER_COUNT < 1:
    raise RuntimeError("MAX_GPU_WORKERS must allow at least one CUDA worker.")
'''
    if gpu_guard not in diagnostics_source:
        raise RuntimeError("Could not locate CUDA guard in diagnostics cell")
    diagnostics_source = diagnostics_source.replace(gpu_guard, replacement_guard)
    diagnostics_source = diagnostics_source.replace(
        '    print("Kaggle dual-T4 environment detected.")\n',
        '    print("Kaggle dual-T4 environment detected.")\nprint(f"GPU evaluation workers: {GPU_WORKER_COUNT}")\n',
    )
    diagnostics_source = diagnostics_source.replace(
        '    warnings.warn("Only one GPU is visible. The notebook supports it, but Kaggle T4 x2 is recommended.")\n',
        '    warnings.warn("Only one GPU is visible; set REQUIRE_DUAL_GPU=False to allow a single-GPU fallback.")\n',
    )
    diagnostics_cell["source"] = source_lines(diagnostics_source)

    helpers_cell = find_cell(notebook, "def display_gpu_memory")
    helpers_source = source_text(helpers_cell)
    old_display = '''def display_gpu_memory():
    for gpu_index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_index)
        allocated = torch.cuda.memory_allocated(gpu_index)
        reserved = torch.cuda.memory_reserved(gpu_index)
        print(
            f"GPU {gpu_index} memory: free={free_bytes / 2**30:.2f} GiB, "
            f"allocated={allocated / 2**30:.2f} GiB, reserved={reserved / 2**30:.2f} GiB, "
            f"total={total_bytes / 2**30:.2f} GiB"
        )
'''
    new_display = '''def display_gpu_memory(gpu_index=None):
    indices = range(torch.cuda.device_count()) if gpu_index is None else [int(gpu_index)]
    for index in indices:
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            allocated = torch.cuda.memory_allocated(index)
            reserved = torch.cuda.memory_reserved(index)
        print(
            f"GPU {index} memory: free={free_bytes / 2**30:.2f} GiB, "
            f"allocated={allocated / 2**30:.2f} GiB, reserved={reserved / 2**30:.2f} GiB, "
            f"total={total_bytes / 2**30:.2f} GiB"
        )
'''
    if old_display not in helpers_source:
        raise RuntimeError("Could not locate GPU memory helper")
    helpers_cell["source"] = source_lines(helpers_source.replace(old_display, new_display))

    loader_cell = find_cell(notebook, "def load_compatible_tokenizer")
    loader_source = source_text(loader_cell)
    old_compatibility_load = '''        warnings.warn(
            f"{model_config['name']}: checkpoint tokenizer was exported by Transformers v5 "
            "as TokenizersBackend. Loading its immutable tokenizer.json through pinned "
            "Transformers v4 PreTrainedTokenizerFast; model files are not modified."
        )
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            str(resolved_path),
            tokenizer_file=str(tokenizer_json_path),
            **common_kwargs,
        )
'''
    new_compatibility_load = '''        print(
            f"TOKENIZER COMPATIBILITY | {model_config['name']}: loading the immutable "
            "Transformers-v5 tokenizer.json through pinned Transformers v4; input files "
            "are not modified."
        )
        with warnings.catch_warnings():
            # Expected because the checkpoint metadata names the v5-only wrapper class.
            warnings.filterwarnings(
                "ignore",
                message=r"The tokenizer class you load from this checkpoint.*",
                category=UserWarning,
            )
            tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
                str(resolved_path),
                tokenizer_file=str(tokenizer_json_path),
                **common_kwargs,
            )
'''
    if old_compatibility_load not in loader_source:
        raise RuntimeError("Could not locate tokenizer compatibility load")
    loader_source = loader_source.replace(old_compatibility_load, new_compatibility_load)
    loader_source = loader_source.replace(
        "def load_model(model_config):\n",
        "def load_model(model_config, device):\n",
    )
    loader_source = loader_source.replace(
        '    print(f"Loading {model_config[\'name\']} from {resolved_path}")\n',
        '    print(f"Loading {model_config[\'name\']} on {device} from {resolved_path}")\n',
    )
    loader_source = loader_source.replace(
        '        device="cuda",\n',
        "        device=device,\n",
    )
    loader_source = loader_source.replace(
        "        parallelize=True,\n",
        "        parallelize=False,\n",
    )
    loader_source = loader_source.replace(
        '        "tokenizer_loading": tokenizer_load_metadata,\n',
        '        "tokenizer_loading": tokenizer_load_metadata,\n        "device": device,\n',
    )
    loader_cell["source"] = source_lines(loader_source)

    runner_cell = find_cell(notebook, "def benchmark_paths")
    runner_source = source_text(runner_cell)
    runner_source = "REQUEST_BUILD_LOCK = threading.Lock()\n\n\n" + OOM_HELPERS + "\n\n" + runner_source
    runner_source = runner_source.replace(
        "            request_groups = prepare_requests(output, model_context)\n",
        "            # Harness request construction mutates task state; serialize only this CPU step.\n"
        "            with REQUEST_BUILD_LOCK:\n"
        "                request_groups = prepare_requests(output, model_context)\n",
    )
    runner_source = runner_source.replace(
        "    display_gpu_memory()\n",
        '    clear_gpu_cache(model_context["gpu_index"])\n'
        '    display_gpu_memory(model_context["gpu_index"])\n'
        '    batch_size = int(BENCHMARK_BATCH_SIZES[benchmark])\n'
        '    chunk_size = int(BENCHMARK_CHUNK_SIZES[benchmark])\n'
        '    if batch_size < 1 or chunk_size < 1:\n'
        '        raise ValueError(f"Invalid batch/chunk size for {benchmark}: {batch_size}/{chunk_size}")\n'
        '    # Never use lm-eval auto batching here: v0.4.9.2 divides by zero on small tail groups.\n'
        '    model_context["lm"].batch_size_per_gpu = batch_size\n'
        '    model_context["lm"].batch_schedule = 1\n'
        '    model_context["lm"].batch_sizes.clear()\n'
        '    print(f"GPU {model_context[\'gpu_index\']} | {benchmark}: batch={batch_size}, chunk={chunk_size}")\n',
    )
    runner_source = runner_source.replace(
        '        desc=f"{model_context[\'model_name\']} | {benchmark}",\n        unit="sample",\n',
        '        desc=f"GPU {model_context[\'gpu_index\']} | {model_context[\'model_name\']} | {benchmark}",\n        unit="sample",\n        position=int(model_context["gpu_index"]),\n        leave=True,\n',
    )
    runner_source = runner_source.replace(
        "            for group_chunk in chunks(pending, SAMPLE_CHUNK_SIZE):\n",
        "            for group_chunk in chunks(pending, chunk_size):\n",
    )
    runner_source = runner_source.replace(
        '                    flat_responses = model_context["lm"].loglikelihood(flat_requests, disable_tqdm=True)\n',
        '                    flat_responses = inference_with_oom_backoff(\n'
        '                        model_context, "loglikelihood", flat_requests\n'
        '                    )\n',
    )
    runner_source = runner_source.replace(
        '                    responses = model_context["lm"].generate_until(flat_requests, disable_tqdm=True)\n',
        '                    responses = inference_with_oom_backoff(\n'
        '                        model_context, "generate_until", flat_requests\n'
        '                    )\n',
    )
    runner_cell["source"] = source_lines(runner_source)

    main_cell = find_cell(notebook, "PIP_FREEZE = subprocess.check_output")
    main_source = source_text(main_cell)
    invocation = "\nvalidate_model_configuration()\n"
    if invocation not in main_source:
        raise RuntimeError("Could not locate the original model-loop invocation")
    prefix = main_source.split(invocation, 1)[0]
    main_cell["source"] = source_lines(prefix + "\n\n" + MAIN_LOOP)

    # IDs should also reflect model-specific replacements.
    clean_notebook(notebook)
    notebook.setdefault("metadata", {}).setdefault("language_info", {})["name"] = "python"
    return notebook


def validate_notebook(notebook: dict, model: dict) -> None:
    serialized = json.dumps(notebook)
    required = [
        model["name"],
        model["path"],
        "ThreadPoolExecutor",
        "GPU_WORKER_COUNT",
        "one_full_model_replica_per_gpu_dynamic_benchmark_queue",
        "parallelize=False",
        "load_compatible_tokenizer",
        "MODELS = [MODEL]",
        "BENCHMARK_BATCH_SIZES",
        "BENCHMARK_CHUNK_SIZES",
        "batch_size_per_gpu = batch_size",
        "inference_with_oom_backoff",
        "PYTORCH_ALLOC_CONF",
    ]
    missing = [item for item in required if item not in serialized]
    if missing:
        raise RuntimeError(f"{model['filename']} is missing required fragments: {missing}")
    forbidden = ["parallelize=True", '"numexpr==2.10.2"', 'DEFAULT_BATCH_SIZE = \\"auto']
    present = [item for item in forbidden if item in serialized]
    if present:
        raise RuntimeError(f"{model['filename']} contains forbidden fragments: {present}")
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    if len(cell_ids) != len(set(cell_ids)):
        raise RuntimeError(f"{model['filename']} contains duplicate cell IDs")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            raise RuntimeError(f"{model['filename']} cell {index} contains stale execution state")
        compile(source_text(cell), f"{model['filename']}::cell-{index}", "exec")


def main() -> None:
    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        notebook = build_notebook(template, model)
        validate_notebook(notebook, model)
        destination = OUTPUT_DIR / model["filename"]
        destination.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {destination.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
