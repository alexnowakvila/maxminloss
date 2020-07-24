#!/usr/bin/env bash

python main.py --task sequence --model crf --add_bias --cython --dataset $1 --reg $2 --check_dual_every 5 --verbose_samples 1000
