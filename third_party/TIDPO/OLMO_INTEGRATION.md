# OLMo integration

The upstream source in this directory is pinned, unmodified source material. The runnable OLMo 2
adapter is `BeeS/olmo2_bees/train_preference_suite.py`, driven by
`olmo_bees_tidpo_simpo_sampo_train_eval.ipynb`.

The adapter retains the imported implementation's gradient-attribution target, gradient/Gaussian
weight mixture, mean-one sequence-weight scaling, TDPO2-adjusted weighted margin, and optional
triplet term. It replaces the resident reference model with a reusable cache. Selected-token
reference log-probabilities are exact. The full-vocabulary position-wise KL is represented by a
documented lower-bound projection that keeps each reference top-k token and merges all remaining
probability mass into one bucket. Fixed, reproducible base-policy anchors replace anchors resampled
from the changing policy at every step. Those bounded changes are necessary for full-parameter OLMo
2 1B training on the local dual-GPU memory budget. SimPO and SamPO share the same data, optimizer,
precision, and FSDP2 architecture for a controlled comparison.

The upstream Python environment is not installed into the workspace: its older PyTorch,
Transformers, and tensor-parallel pins conflict with the already validated OLMo CUDA environment.
