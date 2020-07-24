#!/usr/bin/env bash

python main.py --task ranking --model m4n --add_bias --cython --dataset $1 --reg $2 --check_dual_every 5 --verbose_samples -1 --epochs 51
