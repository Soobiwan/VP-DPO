#!/bin/bash

CUDA_VISIBLE_DEVICES=7 nohup python -u adpo/online/annotate.py > adpo/log/annotate.log 2>&1 &

