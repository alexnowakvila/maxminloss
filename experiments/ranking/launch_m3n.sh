#!/usr/bin/env bash

declare -a StringArray=("glass" "bodyfat" "authorship" "wine" "vowel" "vehicle")


for val in "${StringArray[@]}"
do
    for lambd in 0.3535 0.125 0.0442 0.0156 0.005 0.002 0.0007 0.0002
    do
        if [ $1 -gt 0 ]
        then
            ./experiments/ranking/m3n_cv.sh $val $lambd
        else
            ./experiments/ranking/m3n.sh $val $lambd
        fi
    done
done