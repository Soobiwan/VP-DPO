#!/bin/bash

# TLDR -----------------------------------

# CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch --config_file adpo/config/zero2.yaml --main_process_port 8000 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy random --num_samples 2000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft-tldr --output_dir ./models/model-tldr/tldr-3b-rand-2k \
#                             --dataset_name tldr --run_name tldr-3b-rand-2k \
#                             --learning_rate 1e-6 --beta 0.5 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=2,3 nohup accelerate launch --config_file adpo/config/zero2.yaml --main_process_port 8001 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy ppl_margin --part top --num_samples 2000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft-tldr --output_dir ./models/model-tldr/tldr-3b-ppl-top-2k \
#                             --dataset_name tldr --run_name tldr-3b-ppl-top-2k \
#                             --learning_rate 1e-6 --beta 0.5 >./adpo/log/dpo-top.log &

# CUDA_VISIBLE_DEVICES=4,5 nohup accelerate launch --config_file adpo/config/zero2.yaml --main_process_port 8002 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy ppl_margin --part mid --num_samples 2000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft-tldr --output_dir ./models/model-tldr/tldr-3b-ppl-mid-2k \
#                             --dataset_name tldr --run_name tldr-3b-ppl-mid-2k \
#                             --learning_rate 1e-6 --beta 0.5 >./adpo/log/dpo-mid.log &

# CUDA_VISIBLE_DEVICES=6,7 nohup accelerate launch --config_file adpo/config/zero2.yaml --main_process_port 8003 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy ppl_margin --part bot --num_samples 2000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft-tldr --output_dir ./models/model-tldr/tldr-3b-ppl-bot-2k \
#                             --dataset_name tldr --run_name tldr-3b-ppl-bot-2k \
#                             --learning_rate 1e-6 --beta 0.5 >./adpo/log/dpo-bot.log &

# HH -----------------------------------

# CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch --config_file adpo/config/zero2.yaml adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy random --num_samples 2000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft --output_dir ./models/model-hh/hh-3b-rand-2k \
#                             --dataset_name hh --run_name hh-3b-rand-2k \
#                             --learning_rate 1e-6 --beta 0.1 >./adpo/log/dpo.log &


# CUDA_VISIBLE_DEVICES=5,7 nohup accelerate launch --config_file adpo/config/zero2.yaml --main_process_port 8001 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy ex_reward_margin --part top --num_samples 8000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft --output_dir ./models/model-hh/hh-3b-ex-top-8k \
#                             --dataset_name hh --run_name hh-3b-ex-top-8k \
#                             --learning_rate 1e-6 --beta 0.1 >./adpo/log/dpo-top-hh.log &

# CUDA_VISIBLE_DEVICES=4,5 nohup accelerate launch --config_file adpo/config/zero2.yaml --main_process_port 8002 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy ex_reward_margin --part mid --num_samples 2000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft --output_dir ./models/model-hh/hh-3b-ex-mid-2k \
#                             --dataset_name hh --run_name hh-3b-ex-mid-2k \
#                             --learning_rate 1e-6 --beta 0.1 >./adpo/log/dpo-mid-hh.log &

# CUDA_VISIBLE_DEVICES=6,7 nohup accelerate launch --config_file adpo/config/zero2.yaml --main_process_port 8003 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy ex_reward_margin --part bot --num_samples 2000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft --output_dir ./models/model-hh/hh-3b-ex-bot-2k \
#                             --dataset_name hh --run_name hh-3b-ex-bot-2k \
#                             --learning_rate 1e-6 --beta 0.1 >./adpo/log/dpo-bot-hh.log &




# ultrafeedback -----------------------------------

# CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch --config_file adpo/config/zero2.yaml adpo/dpo.py \
                            # --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
                            # --strategy random --num_samples 2000 \
                            # --model_name ./models/model/Llama-3.2-3B-sft --output_dir ./models/model-uf/uf-3b-rand-2k \
                            # --dataset_name uf --run_name uf-3b-rand-2k \
                            # --learning_rate 1e-6 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=5,7 nohup accelerate launch --config_file adpo/config/zero2.yaml --main_process_port 8001 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy ex_reward_margin --part top --num_samples 6000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft --output_dir ./models/model-uf/uf-3b-ex-top-6k \
#                             --dataset_name uf --run_name uf-3b-ex-top-6k \
#                             --learning_rate 1e-6 --beta 0.1 >./adpo/log/dpo-top-uf.log &

# CUDA_VISIBLE_DEVICES=4,5 nohup accelerate launch --config_file adpo/config/zero2.yaml --main_process_port 8002 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy ppl_margin --part mid --num_samples 2000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft --output_dir ./models/model-uf/uf-3b-ppl-mid-2k \
#                             --dataset_name uf --run_name uf-3b-ppl-mid-2k \
#                             --learning_rate 1e-6 --beta 0.1 >./adpo/log/dpo-mid-uf.log &

# CUDA_VISIBLE_DEVICES=6,7 nohup accelerate launch --config_file adpo/config/zero2.yaml --main_process_port 8003 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy ppl_margin --part bot --num_samples 2000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft --output_dir ./models/model-uf/uf-3b-ppl-bot-2k \
#                             --dataset_name uf --run_name uf-3b-ppl-bot-2k \
#                             --learning_rate 1e-6 --beta 0.1 >./adpo/log/dpo-bot-uf.log &

# uf 8b -----------------------------------

# CUDA_VISIBLE_DEVICES=1,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy random --num_samples -1 \
#                             --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-uf/uf-8b-dpo-full \
#                             --dataset_name uf --run_name uf-8b-dpo-full \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=1,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8101 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
#                             --strategy random --num_samples -1 \
#                             --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-abl/uf-8b-dpo-full-bs128 \
#                             --dataset_name uf --run_name uf-8b-dpo-full-bs128 \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo1.log &
# wait
# CUDA_VISIBLE_DEVICES=1,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8102 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
#                             --strategy random --num_samples -1 \
#                             --model_name ./models/model/Llama-3-8B-sft --output_dir ./models/model-abl/uf-8b-sft-dpo-full-bs128 \
#                             --dataset_name uf --run_name uf-8b-sft-dpo-full-bs128 \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo2.log &
# wait
# CUDA_VISIBLE_DEVICES=1,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8103 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
#                             --strategy random --num_samples -1 \
#                             --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-abl/lla-uf-8b-dpo-full-bs128 \
#                             --dataset_name llama_uf --run_name lla-uf-8b-dpo-full-bs128 \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo3.log &
# wait

# CUDA_VISIBLE_DEVICES=1,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8104 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
#                             --strategy random --num_samples -1 \
#                             --model_name ./models/model/Llama-3-8B-sft --output_dir ./models/model-abl/lla-uf-8b-sft-dpo-full-bs128 \
#                             --dataset_name llama_uf --run_name lla-uf-8b-sft-dpo-full-bs128 \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo4.log &
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy im_reward_margin --part top --num_samples 10000 \
#                             --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-uf/uf-8b-im-top-10k \
#                             --dataset_name uf --run_name uf-8b-im-top-10k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo_1.log &

# CUDA_VISIBLE_DEVICES=2,3,4,6 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy ppl_margin --part mid --num_samples 6000 \
#                             --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-ppl/uf-8b-ppl-mid-6k \
#                             --dataset_name uf --run_name uf-8b-ppl-mid-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# uf my-8b-sft -----------------------------------

# CUDA_VISIBLE_DEVICES=0,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy random --num_samples -1 \
#                             --model_name ./models/model/Llama-3-8B-sft --output_dir ./models/model-uf/uf-8b-sft-dpo-full \
#                             --dataset_name uf --run_name uf-8b-sft-dpo-full \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy random --num_samples 6000 \
#                             --model_name ./models/model/Llama-3-8B-sft --output_dir ./models/model-uf/uf-8b-sft-dpo-rand-6k \
#                             --dataset_name uf --run_name uf-8b-sft-dpo-rand-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy ex_reward_margin --part top --num_samples 6000 \
#                             --model_name ./models/model/Llama-3-8B-sft --output_dir ./models/model-uf/uf-8b-sft-dpo-ex-top-6k \
#                             --dataset_name uf --run_name uf-8b-sft-dpo-ex-top-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8002 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy im_reward_margin --part top --num_samples 6000 \
#                             --model_name ./models/model/Llama-3-8B-sft --output_dir ./models/model-uf/uf-8b-sft-dpo-im-top-6k \
#                             --dataset_name uf --run_name uf-8b-sft-dpo-im-top-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo_.log &

# CUDA_VISIBLE_DEVICES=2,3,4,6 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8002 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy ppl_margin --part mid --num_samples 6000 \
#                             --model_name ./models/model/Llama-3-8B-sft --output_dir ./models/model-ppl/uf-8b-sft-dpo-ppl-mid-6k \
#                             --dataset_name uf --run_name uf-8b-sft-dpo-ppl-mid-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# llama_uf -----------------------------------

# CUDA_VISIBLE_DEVICES=1,2 nohup accelerate launch --config_file adpo/config/zero2.yaml --main_process_port 8001 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy random --num_samples 2000 \
#                             --model_name ./models/model/Llama-3.2-3B-sft --output_dir ./models/model-uf-lla/lla-uf-3b-rand-2k \
#                             --dataset_name llama_uf --run_name lla-uf-3b-rand-2k \
#                             --learning_rate 1e-6 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8002 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy random --num_samples 6000 \
#                             --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-uf-lla/lla-uf-8b-rand-6k \
#                             --dataset_name llama_uf --run_name lla-uf-8b-rand-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo_.log &

# CUDA_VISIBLE_DEVICES=2,3,4,6 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8003 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy ppl_margin --part mid --num_samples 6000 \
#                             --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-ppl/lla-uf-8b-ppl-mid-6k \
#                             --dataset_name llama_uf --run_name lla-uf-8b-ppl-mid-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8002 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy random --num_samples 6000 \
#                             --model_name ./models/model/Llama-3-8B-sft --output_dir ./models/model-uf-lla/lla-uf-8b-sft-dpo-rand-6k \
#                             --dataset_name llama_uf --run_name lla-uf-8b-sft-dpo-rand-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &


# CUDA_VISIBLE_DEVICES=3,4,5,6 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy mul --part top --num_samples 6000 \
#                             --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-uf-lla/lla-uf-8b-mul-top-6k-ep1 \
#                             --dataset_name llama_uf --run_name lla-uf-8b-mul-top-6k-ep1 \
#                             --learning_rate 3e-7 --beta 0.01 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8003 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy add --part top --num_samples 3000 \
#                             --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-uf-lla/lla-uf-8b-add-top-3k-ep1 \
#                             --dataset_name llama_uf --run_name lla-uf-8b-add-top-3k-ep1 \
#                             --learning_rate 4e-7 --beta 0.01 >./adpo/log/dpo.log &
# echo "wait for 20 minutes"
# sleep 1200
# echo "start dpo training"

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8013 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy add --part top --num_samples 10000 \
#                             --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-uf-lla/lla-uf-8b-add-top-10k-ep1 \
#                             --dataset_name llama_uf --run_name lla-uf-8b-add-top-10k-ep1 \
#                             --learning_rate 4e-7 --beta 0.01 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8083 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy mul --part top --num_samples 6000 \
#                             --model_name /NAS/dengx/models/model/Qwen2.5-7B-Instruct --output_dir ./models/model-uf-lla/lla-uf-qwen-7b-mul-top-6k-ep1 \
#                             --dataset_name llama_uf --run_name lla-uf-qwen-7b-mul-top-6k-ep1 \
#                             --learning_rate 5e-7 --beta 0.01 >./adpo/log/dpo.log &
# wait

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8093 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy random --num_samples -1 \
#                             --model_name /NAS/dengx/models/model/Qwen2.5-7B-Instruct --output_dir ./models/model-uf-lla/lla-uf-qwen-7b-full-ep1 \
#                             --dataset_name llama_uf --run_name lla-uf-qwen-7b-full-ep1 \
#                             --learning_rate 5e-7 --beta 0.01 >./adpo/log/dpo_.log &

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8073 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy mul --part top --num_samples 6000 \
#                             --model_name /NAS/dengx/models/model/Qwen2.5-14B-Instruct --output_dir ./models/model-uf-lla/lla-uf-qwen-14b-mul-top-6k-ep2 \
#                             --dataset_name llama_uf --run_name lla-uf-qwen-14b-mul-top-6k-ep2 \
#                             --learning_rate 5e-7 --beta 0.01 >./adpo/log/dpo.log &

# wait

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8073 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy add --part top --num_samples 15000 \
#                             --model_name /NAS/dengx/models/model/Qwen2.5-14B-Instruct --output_dir ./models/model-uf-lla/lla-uf-qwen-14b-add-top-15k-ep1 \
#                             --dataset_name llama_uf --run_name lla-uf-qwen-14b-add-top-15k-ep1 \
#                             --learning_rate 5e-7 --beta 0.01 --num_train_epochs 1 >./adpo/log/dpo.log &
# wait

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8093 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy random --num_samples -1 \
#                             --model_name /NAS/dengx/models/model/Qwen2.5-14B-Instruct --output_dir ./models/model-uf-lla/lla-uf-qwen-14b-full-ep1 \
#                             --dataset_name llama_uf --run_name lla-uf-qwen-14b-full-ep1 \
#                             --learning_rate 5e-7 --beta 0.01 >./adpo/log/dpo_.log &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy ex_reward_margin --part top --num_samples 6000 \
#                             --model_name ./models/model/Llama-3-8B-sft --output_dir ./models/model-uf-lla/lla-uf-8b-sft-dpo-ex-top-6k \
#                             --dataset_name llama_uf --run_name lla-uf-8b-sft-dpo-ex-top-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=2,3,4,6 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8004 adpo/dpo.py \
                            # --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
                            # --strategy ppl_margin --part mid --num_samples 6000 \
                            # --model_name ./models/model/Llama-3-8B-sft --output_dir ./models/model-ppl/lla-uf-8b-sft-dpo-ppl-mid-6k \
                            # --dataset_name llama_uf --run_name lla-uf-8b-sft-dpo-ppl-mid-6k \
                            # --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &


# mistral_uf -----------------------------------

# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8002 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy random --num_samples -1 \
#                             --model_name ./models/model/Mistral-7B-Instruct-v2 --output_dir ./models/model-uf-mis/mis-uf-7B-full \
#                             --dataset_name mistral_uf --run_name mis-uf-7B-full \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8002 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy random --num_samples 6000 \
#                             --model_name ./models/model/Mistral-7B-Instruct-v2 --output_dir ./models/model-uf-mis/mis-uf-7B-rand-6k \
#                             --dataset_name mistral_uf --run_name mis-uf-7B-rand-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8002 adpo/dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy ex_reward_margin --part top --num_samples 6000 \
#                             --model_name ./models/model/Mistral-7B-Instruct-v2 --output_dir ./models/model-uf-mis/mis-uf-7B-ex-top-6k \
#                             --dataset_name mistral_uf --run_name mis-uf-7B-ex-top-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=2,3,4,6 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8005 adpo/dpo.py \
#                             --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#                             --strategy ppl_margin --part mid --num_samples 6000 \
#                             --model_name ./models/model/Mistral-7B-Instruct-v2 --output_dir ./models/model-ppl/mis-uf-7b-ppl-mid-6k \
#                             --dataset_name mistral_uf --run_name mis-uf-7b-ppl-mid-6k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# llama_uf_armo
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/dpo.py \
#                     --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                     --strategy in_reward_margin --part top --num_samples 20000 \
#                     --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-uf-lla/armo-8b-mix-top-20k-ep1-4e-7 \
#                     --dataset_name llama_uf_armo --run_name armo-8b-mix-top-20k-ep1-4e-7 \
#                     --learning_rate 4e-7 --beta 0.01 --num_train_epochs 1 >./adpo/log/dpo.log &

# iterative dpo -----------------------------------

# r0

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/iterative_dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --model_name ./models/model/Llama-3-8B-sft --output_dir ./models/model-iter/r0-top-5k \
#                             --dataset_name r0 --run_name r0-top-5k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# CUDA_VISIBLE_DEVICES=0,1,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/iterative_dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy random --num_samples 5000 \
#                             --model_name ./models/model/Meta-Llama-3-8B-Instruct --output_dir ./models/model-iter-ins/r0-top-5k \
#                             --dataset_name r0 --run_name r0-top-5k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# r1


# CUDA_VISIBLE_DEVICES=3,4,5,6 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/iterative_dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --model_name models/model-iter/r0-full/checkpoint-1250 --output_dir ./models/model-iter/r1-top-5k \
#                             --dataset_name r1 --run_name r1-top-5k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo_.log &

# CUDA_VISIBLE_DEVICES=0,1,6,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/iterative_dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy random --num_samples 4000 \
#                             --model_name models/model-iter-ins/r0-full/checkpoint-1250 --output_dir ./models/model-iter-ins/r1-top-4k \
#                             --dataset_name r1 --run_name r1-top-4k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &

# r2

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/iterative_dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy random --num_samples -1 \
#                             --model_name models/model-iter/r1-full/checkpoint-1250 --output_dir ./models/model-iter/r2-full \
#                             --dataset_name r2 --run_name r2-full \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo_.log &



# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8002 adpo/iterative_dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy ex_reward_margin --num_samples 5000 \
#                             --model_name models/model-iter/r1-full/checkpoint-1250 --output_dir ./models/model-iter/r2-top-5k \
#                             --dataset_name r2 --run_name r2-top-5k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo_.log &


# CUDA_VISIBLE_DEVICES=0,1,4,7 nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/iterative_dpo.py \
#                             --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
#                             --strategy random --num_samples 4000 \
#                             --model_name models/model-iter-ins/r1-top-5k/checkpoint-312 --output_dir ./models/model-iter-ins/r2-top-4k \
#                             --dataset_name r2 --run_name r2-top-4k \
#                             --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &