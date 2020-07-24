#!/usr/bin/env bash

declare -a StringArray=("wisconsin" "stocks" "machinecpu" "abalone" "auto")

for lambd in 0.3535 0.125 0.0442 0.0156 0.005 0.002 0.0007 0.0002 0.00008
do

    for val in "${StringArray[@]}"
    do
        if [ $1 -gt 0 ]
        then
            ./experiments/ordinal/crf_cv.sh $val $lambd
        else
            ./experiments/ordinal/crf.sh $val $lambd
        fi
    done
done