#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 nohup python adpo/online/gen.py --model_name_or_path models/model/Llama-3-8B-sft --round 0 > adpo/log/gen.log &

# CUDA_VISIBLE_DEVICES=4 nohup python adpo/online/gen.py --model_name_or_path models/model-iter/r0-full/checkpoint-1250 --round 1 > adpo/log/gen.log &

# CUDA_VISIBLE_DEVICES=2 nohup python adpo/online/gen.py --model_name_or_path models/model-iter/r1-full/checkpoint-1250 --round 2 > adpo/log/gen.log &