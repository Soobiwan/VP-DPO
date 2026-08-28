#!/bin/bash

source ~/.bashrc
eval "$(conda shell.bash hook)"

# 定义可用的 GPU 索引数组
available_gpus=(0 1 2 3)  # 可以根据实际情况修改这个数组
data_path="dataset/online-ins-full/"

for round in {0..2}
do
    echo "Starting round ${round}"

    if [ $round -eq 0 ]; then
        model_path="models/model/Meta-Llama-3-8B-Instruct"
    else
        prev_round=$((round-1))
        model_path="models/model-iter-ins/r${prev_round}-full/checkpoint-625"
    fi
    conda activate te_llm
    # Generation phase
    for idx in "${!available_gpus[@]}"
    do
        gpu=${available_gpus[$idx]}
        echo "Running generation on GPU ${gpu}"
        CUDA_VISIBLE_DEVICES=${gpu} nohup python adpo/online/gen.py \
            --model_name_or_path ${model_path} \
            --output_dir ${data_path} \
            --round ${round} \
            --local_index ${idx} \
            > adpo/log/gen${round}_${idx}.log &
    done

    wait

    # Annotation phase
    for idx in "${!available_gpus[@]}"
    do
        gpu=${available_gpus[$idx]}
        echo "Running annotation on GPU ${gpu}"
        CUDA_VISIBLE_DEVICES=${gpu} nohup python -u adpo/online/annotate.py \
            --generation_file ${data_path}round${round}-idx${idx}.json \
            --output_dir ${data_path}r${round}/${idx}/ \
            > adpo/log/annotate${round}_${idx}.log 2>&1 &
    done

    wait
    echo "Completed round ${round}"
    conda activate tr_llm
    CUDA_VISIBLE_DEVICES=${available_gpus[0]},${available_gpus[1]},${available_gpus[2]},${available_gpus[3]} nohup accelerate launch --config_file adpo/config/zero3.yaml --main_process_port 8001 adpo/iterative_dpo.py \
                            --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
                            --strategy random --num_samples -1 \
                            --model_name ${model_path} --output_dir ./models/model-iter-ins/r${round}-full \
                            --dataset_name r${round}  --dataset_path ${data_path} --run_name r${round}-full \
                            --learning_rate 5e-7 --beta 0.1 >./adpo/log/dpo.log &
    wait
done

echo "All rounds completed"