#!/usr/bin/env bash

python main.py --task ordinal --model m4n --add_bias --cython --dataset $1 --reg $2 --check_dual_every 10 --iter_oracle 20 --kernel --epochs 101