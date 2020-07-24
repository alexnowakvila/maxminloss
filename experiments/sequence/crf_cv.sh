#!/usr/bin/env bash

python main_cv.py --task sequence --model crf --add_bias --cython --dataset $1 --reg $2 --check_dual_every 5 --verbose_samples -1 --n_samples 2000 --epochs 21 --kernel