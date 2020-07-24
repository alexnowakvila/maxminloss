#!/usr/bin/env bash

python main_oracle.py --task ordinal --model m4n --add_bias --cython --dataset $1 --reg $2 --check_dual_every 10 --iter_oracle $3 --iter_oracle_log 500 --kernel --epochs $4 --warmstart $5