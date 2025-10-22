# DPO data selection

This repository contains implementation of the paper: [Less is More: Improving LLM Alignment via Preference Data Selection][paper-link].

## Features

- Standard SFT, DPO, IPO, KTO training implementation
- Iterative DPO with dynamic data selection
- Comprehensive evaluation pipeline
- Reward calculation
- Support for multiple model architectures (LLaMA, Mistral, Qwen)

## Project Structure

Note that there are a `models` and `datasets` directory at the same level as `adpo`: I downloaded the related models and datasets in these two files, and stored the generated checkpoints and datas in these two directories. But you can choose not to use it if the model/dataset loading is fast in your linux machine. And remember to change the dataset/model loading code in the dpo.py, sft.py corerspondingly.

```
models
datasets
adpo
├── config/           # Configuration files for different models and training setups
├── eval/            # Evaluation scripts and templates
├── metric/          # Metric calculation and visualization tools
├── online/          # Online iterative DPO implementation
├── template/        # Prompt templates for different tasks
└── core scripts:
    ├── dpo.py      # Main DPO training implementation
    ├── sft.py      # Supervised fine-tuning implementation
    ├── rm.py       # External reward calculation
    ├── ifd.py      # Implicit feedback calculation
    ├── m_sampler.py # margin fusion and data selection
    └── sampler.py  # Data sampling and selection

```

## Installation

This project requires two separate Python environments: one for training and one for evaluation. You can refer to the [RLHF-Reward-Modeling repository][env-setup] for reference. There may be some minor bug when you create the env as I have not use these code for about 4 months and everything is changing fast.

### Training Environment

```bash
# Create and activate training environment
conda create -n adpo-train python=3.10.9
conda activate adpo-train

# Install required packages
pip install trl==0.9.6
pip install flash-attn==2.7.4.post1
pip install accelerate==1.4.0
pip install transformers==4.49.0
```

### Evaluation Environment

```bash
# Create and activate evaluation environment
conda create -n adpo-eval python=3.10.9
conda activate adpo-eval

# Install required packages
pip install torch==2.4.0
pip install transformers==4.49.0
pip install vllm==0.6.1
```

## Usage

### Standard DPO Training

```bash
# Run DPO training with default configuration
bash adpo/dpo.sh
```

### Online DPO

```bash
# Start online DPO pipeline
bash adpo/online/run.sh
```

### Evaluation

```bash
# Run evaluation on generated outputs
python adpo/eval/alpaca.py # for alpaca_eval2.0
```

```bash
# Run evaluation on generated outputs
python adpo/eval/vllm_gen.py  
python parallel_eval_oai.py   # for tl;dr, hh, alpaca
```


## Metrics and Visualization
rm.py and ifd.py calculates rewards and margins and the results are stored in `adpo/metric/` directory.
The `metric/` directory contains scripts for calculating and visualizing various metrics:
- Reward distribution analysis
- PPL (perplexity) tracking
- Online reward plotting
- 2D distribution visualization

## Configuration

Model and training configurations are stored in YAML files under the `config/` directory:
- `llama3-8b-vllm-gen.yaml`: Configuration for LLaMA-3 8B model
- `mistral-7b-vllm-gen.yaml`: Configuration for Mistral 7B model
- `qwen-7b-vllm-gen.yaml`: Configuration for Qwen 7B model
- `zero2.yaml` and `zero3.yaml`: DeepSpeed ZeRO stage configurations


[paper-link]: https://arxiv.org/pdf/2502.14560
[env-setup]: https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/bradley-terry-rm