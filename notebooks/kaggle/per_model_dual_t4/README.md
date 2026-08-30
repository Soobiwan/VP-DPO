# Per-model dual-T4 evaluation notebooks

Each notebook in this folder evaluates exactly one model, computes all six local benchmark
scores, and exports a model-specific ZIP from `/kaggle/working`. Run one notebook at a time.

## Kaggle setup

1. Create/import the desired notebook in Kaggle and attach the model dataset named in its
   configuration cell.
2. Select the **GPU T4 x2** accelerator and enable Internet.
3. Add a Kaggle secret named `Huggingface`. Its token must have access to the accepted GPQA
   dataset terms.
4. Run all cells. The two GPUs load independent replicas of the same 1B model and pull
   benchmarks from a shared queue. MMLU and IFEval are queued first so both GPUs start with
   substantial work.
5. Download the ZIP path printed by the final cell before ending the Kaggle session.

Raw benchmark records are appended in small chunks. A benchmark is skipped on rerun only when
its `DONE` marker, sample inventory, and SHA-256 digest all validate, so an interrupted run can
resume safely within the same Kaggle working directory. A failed benchmark does not stop the
other GPU from completing the remaining queue.

The notebooks use fixed per-benchmark batch sizes. This avoids the pinned harness's
`auto:4` tail-batch divide-by-zero. IFEval and GSM8K submit only four examples per outer call,
so their visible progress and durable JSONL checkpoint advance after each four-example batch
instead of waiting for a sixteen-example generation chunk. IFEval retains the official
1,280-token generation ceiling; reducing that ceiling would make its score a different protocol.

## Notebook index

| # | Model | Notebook |
|---:|---|---|
| 1 | VPDPO_B_Norm_DPO | `01_vpdpo_b_norm_dpo_dual_t4.ipynb` |
| 2 | VPDPO_B_Norm_VDPO | `02_vpdpo_b_norm_vdpo_dual_t4.ipynb` |
| 3 | VPDPO_B_DPO | `03_vpdpo_b_dpo_dual_t4.ipynb` |
| 4 | VPDPO_B_VDPO | `04_vpdpo_b_vdpo_dual_t4.ipynb` |
| 5 | VPDPO_C_DPO | `05_vpdpo_c_dpo_dual_t4.ipynb` |
| 6 | VPDPO_C_VDPO | `06_vpdpo_c_vdpo_dual_t4.ipynb` |
| 7 | Simple_DPO | `07_simple_dpo_dual_t4.ipynb` |
| 8 | VPDPO_A | `08_vpdpo_a_dual_t4.ipynb` |
| 9 | SimPO | `09_simpo_dual_t4.ipynb` |
| 10 | SAMPO | `10_sampo_dual_t4.ipynb` |
| 11 | TIDPO | `11_tidpo_dual_t4.ipynb` |

## Failure fixes carried by every notebook

- A Transformers-v5 checkpoint declaring `TokenizersBackend` is loaded from its immutable
  `tokenizer.json` through pinned Transformers v4, then passed to `HFLM` as an already-created
  tokenizer. The input checkpoint is never modified. The expected class-name warning is
  suppressed only after an exact tokenizer-ID probe succeeds.
- `parallelize=True` was removed. That option model-sharded a small model over both GPUs while
  still running benchmarks serially.
- The harness's unstable `auto:4` batching was replaced with fixed per-benchmark T4 batch sizes;
  this fixes the observed MMLU `ZeroDivisionError` on a small tail group.
- Likelihood batches now start at 8 instead of 16. Any remaining CUDA OOM automatically clears
  traceback-held tensors and the CUDA cache, halves the active batch, and retries the same
  uncommitted chunk down to batch size 1.
- IFEval/GSM8K generation chunks were reduced from 16 to 4 so progress and resumable artifacts
  update promptly without changing the benchmark's official output-length allowance.
- The forced `numexpr==2.10.2` downgrade was removed because it conflicts with Kaggle's current
  `blosc2` environment and is not needed by the evaluation path.
- Saved execution output and old tracebacks were removed from every generated notebook.

To regenerate the notebooks after changing the shared suite source, run:

```powershell
python .\scripts\build_dual_t4_eval_notebooks.py
```
