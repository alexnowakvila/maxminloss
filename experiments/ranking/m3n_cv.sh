#!/usr/bin/env bash

python main_cv2.py --task ranking --model m3n --add_bias --cython --dataset $1 --reg $2 --check_dual_every 10 --verbose_samples -1 --epochs 51 --kernel