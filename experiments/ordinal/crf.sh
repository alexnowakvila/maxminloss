#!/usr/bin/env bash

python main.py --task ordinal --model crf --add_bias --cython --dataset $1 --reg $2 --check_dual_every 1 --kernel