#!/bin/bash

# sft on openhermes

# CUDA_VISIBLE_DEVICES=4,6 nohup accelerate launch --config_file adpo/config/zero2.yaml ./adpo/sft.py --model_name ./models/Llama-3.2-1B --output_dir ./adpo/model/Llama-3.2-1B-sft --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --run_name 1b-sft-20w >>./adpo/log/sft.log &

# CUDA_VISIBLE_DEVICES=4,5,7 nohup accelerate launch --config_file adpo/config/zero2.yaml ./adpo/sft.py --model_name ./models/Llama-3.2-3B --output_dir ./adpo/model/Llama-3.2-3B-sft --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --run_name 3b-sft-20w >>./adpo/log/sft1.log &

# CUDA_VISIBLE_DEVICES=5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml ./adpo/sft.py --model_name ./models/meta-llama-3-8b --output_dir ./adpo/model/Llama-3-8b-sft --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --run_name 8b-sft-40w >>./adpo/log/sft.log &

# sft on domain data
CUDA_VISIBLE_DEVICES=5,1 nohup accelerate launch --config_file adpo/config/zero3.yaml ./adpo/domain-sft.py --model_name ./adpo/models/model/Llama-3.2-1B-sft --output_dir ./adpo/models/model-s/Llama-3.2-1B-sft-tldr-few --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --run_name 1b-sft-tldr-few >>./adpo/sft.log &

# CUDA_VISIBLE_DEVICES=0,6 nohup accelerate launch --config_file adpo/config/zero2.yaml ./adpo/domain-sft.py --model_name ./adpo/model/Llama-3.2-3B-sft --output_dir ./adpo/model-s/Llama-3.2-3B-sft-tldr-few --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --run_name 3b-sft-tldr >>./adpo/log/sft.log &

# CUDA_VISIBLE_DEVICES=5,6,7 nohup accelerate launch --config_file ./adpo/config/zero3.yaml ./adpo/domain-sft.py --model_name ./adpo/model/Llama-3-8B-sft --output_dir ./adpo/model-s/Llama-3-8B-sft-tldr-few --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --run_name 8b-sft-tldr >>./adpo/sft.log &
