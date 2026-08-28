# BeeS-selected UltraFeedback DPO for OLMo 2 1B

The two notebooks in `notebooks/` implement the complete local workflow:

1. `01_create_bees_ultrafeedback_olmo2.ipynb`
   - downloads and pins UltraFeedback Binarized;
   - converts it to TRL's conversational preference schema;
   - removes (rather than truncates) pairs that exceed the configured OLMo token budget;
   - trains a full-parameter OLMo 2 1B SFT proxy on a deterministic 2,000-pair seed;
   - scores implicit reward margins on both GPUs with resumable JSONL shards;
   - combines implicit margins with the dataset's independent GPT-4 judge-score margins using
     BeeS Eq. (3), then saves the top positive-margin pairs.

2. `02_train_olmo2_1b_dpo.ipynb`
   - starts from `allenai/OLMo-2-0425-1B-SFT`;
   - trains every parameter with DPO (no PEFT, LoRA, QLoRA, or weight quantization);
   - full-shards parameters and gradients across two FSDP processes so both GPUs compute;
   - keeps FP32 master parameters/updates and uses loss-scaled FP16 only for compute because RTX
     20-series GPUs do not support BF16;
   - uses `bitsandbytes.optim.PagedAdamW32bit`, activation checkpointing/offloading, and
     precomputed reference log-probabilities to fit the 8 GB GPU;
   - evaluates held-out preferences and a mandatory five-task `lm-eval` suite, and refuses to mark the
   model approved if the configured regression gate fails.

The workspace root also contains two experiment notebooks built on the same prepared segmented
split and memory-validated OLMo/FSDP2 path:

- `olmo_bees_all_methods_dual_gpu.ipynb` trains the five segment-structured variants.
- `olmo_bees_tidpo_simpo_sampo_train_eval.ipynb` trains TIDPO, SimPO, and SamPO sequentially from
  the same pinned base model, then runs the shared held-out and `lm-eval` comparison. The imported
  TIDPO source and integration notes live under `third_party/TIDPO`.

All downloads, temporary files, package caches, datasets, checkpoints, and evaluation results are
redirected into the parent workspace (`VPDPO/.cache`, `VPDPO/.tmp`, and `VPDPO/artifacts`).

Intermediate FSDP checkpoints deliberately contain a full restart state plus one paged-Adam shard
per GPU. At most two are retained. Budget roughly 55–65 GiB per training run for two restartable
checkpoints and the final model; the proxy and final runs can therefore use roughly 120 GiB together.
`--resume` restores model, scheduler, scaler, RNG, and both paged optimizer shards. Keep the GPU
order fixed when resuming because those optimizer shards are rank-local.

## Validation on this workstation

- Imported BeeS commit: `749faf478e2827dd72835d574693623926a2e444`.
- Pinned UltraFeedback revision: `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`.
- Pinned OLMo revision: `0d85a3d037876ce6ac7d4311d994400fc66ac27f`.
- Full preparation completed: 57,589 losslessly fitting train pairs, 1,891 test pairs, and a
  deterministic 2,000-pair proxy seed.
- A real two-GPU OLMo run used four actual 1,024-token pairs, kept finite gradients on both update
  steps, and reached a measured 7.525 GiB peak reservation on the smaller GPU. A separate run
  successfully resumed a paged-AdamW32 FSDP checkpoint. The real two-model implicit scorer also
  completed on both GPUs with exact row coverage.

## Important accuracy statement

No training recipe can guarantee an accuracy improvement before evaluation. This workflow protects
the original SFT checkpoint, writes DPO checkpoints to a new directory, avoids lossy model
compression, uses conservative published DPO settings, and provides a fail-closed comparison gate.
Only a checkpoint that passes the held-out preference and general benchmark checks should be used as
the approved model.
