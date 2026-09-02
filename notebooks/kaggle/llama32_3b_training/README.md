# Llama 3.2 3B BeeS training notebooks

These notebooks train one non-TIDPO preference method per Kaggle **GPU T4 x2** session
from `meta-llama/Llama-3.2-3B-Instruct` at pinned revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`. TIDPO is deliberately
excluded while its original notebook is under correction.

All notebooks use the same canonical BeeS file:
`data/processed/ultrafeedback_bees_olmo2_1b_segmented_final.jsonl` (LF-normalized SHA-256 `6cddda3cedd1c078ba1f2cc3c3e798d5eeb79968478730b593a206c8ff4eb013`). The old `olmo2` filename
is dataset provenance only; every notebook retokenizes the unchanged text and segment
annotations with the pinned Llama tokenizer.

## Kaggle setup

1. Select **GPU T4 x2** and enable Internet.
2. Accept Meta's Llama 3.2 license on Hugging Face.
3. Add a Kaggle secret named `Huggingface` or `HF_TOKEN`.
4. Attach the repository/dataset input, or allow the notebook to clone the repository.
5. Run one notebook at a time. Objectives requiring a reference cache can be split into a
   reference-only version followed by a training version.

Standalone notebook uploads stage a writable copy of the training package and apply the
Llama token-count/remounted-cache compatibility changes there. The attached or cloned
repository and canonical BeeS dataset are never modified.

## Notebook index

| # | Method | Notebook |
|---:|---|---|
| 1 | VPDPO B_norm-DPO | `01_llama32_3b_b_norm_dpo_kaggle_t4x2.ipynb` |
| 2 | VPDPO B_norm-VDPO | `02_llama32_3b_b_norm_vdpo_kaggle_t4x2.ipynb` |
| 3 | VPDPO B-DPO | `03_llama32_3b_b_dpo_kaggle_t4x2.ipynb` |
| 4 | VPDPO B-VDPO | `04_llama32_3b_b_vdpo_kaggle_t4x2.ipynb` |
| 5 | VPDPO C-DPO | `05_llama32_3b_c_dpo_kaggle_t4x2.ipynb` |
| 6 | VPDPO C-VDPO | `06_llama32_3b_c_vdpo_kaggle_t4x2.ipynb` |
| 7 | Simple DPO | `07_llama32_3b_simple_dpo_kaggle_t4x2.ipynb` |
| 8 | VPDPO A | `08_llama32_3b_vpdpo_a_kaggle_t4x2.ipynb` |
| 9 | SimPO | `09_llama32_3b_simpo_kaggle_t4x2.ipynb` |
| 10 | SamPO | `10_llama32_3b_sampo_kaggle_t4x2.ipynb` |

The recipe is full-parameter FSDP2 training with FP32 master weights and paged FP32 AdamW
state, FP16 compute, no LoRA/PEFT/QLoRA, no quantization, and no restart checkpoints. The
3B recipe is materially heavier than the 1B recipe; use the split reference phase and
preserve the final-only output policy. These generated notebooks have static validation,
but must still be runtime-validated on Kaggle's current image before results are reported.

Regenerate both model-size suites from the repository root with:

```powershell
python .\BeeS\tools\build_kaggle_llama32_training_notebooks.py
```
