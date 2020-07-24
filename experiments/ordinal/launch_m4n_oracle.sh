#!/usr/bin/env bash

for it in 1 5 10 20 50 100
do

    ./experiments/ordinal/m4n_oracle.sh abalone 7.00E-04 $it 201 0
    # ./experiments/ordinal/m4n_oracle.sh wisconsin 1.25E-01 $it 201 0
    # ./experiments/ordinal/m4n_oracle.sh stocks 7.00E-04 $it 201 0
    # ./experiments/ordinal/m4n_oracle.sh machinecpu 4.42E-02 $it 201 0
    # ./experiments/ordinal/m4n_oracle.sh auto 5.00E-03 $it 201 0

    ./experiments/ordinal/m4n_oracle.sh abalone 7.00E-04 $it 201 1
    # ./experiments/ordinal/m4n_oracle.sh wisconsin 1.25E-01 $it 201 1
    # ./experiments/ordinal/m4n_oracle.sh stocks 7.00E-04 $it 201 1
    # ./experiments/ordinal/m4n_oracle.sh machinecpu 4.42E-02 $it 201 1
    # ./experiments/ordinal/m4n_oracle.sh auto 5.00E-03 $it 201 1
done