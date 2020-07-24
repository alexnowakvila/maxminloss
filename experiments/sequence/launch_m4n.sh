#!/usr/bin/env bash

declare -a StringArray=("ocr" "conll" "pos" "ner")

for val in "${StringArray[@]}"
do
    for lambd in 0.0156 0.005 0.002 0.0007 0.0002
    do
        if [ $1 -gt 0 ]
            then
                ./experiments/sequence/m4n_cv.sh $val $lambd
            else
                ./experiments/sequence/m4n.sh $val $lambd
            fi
    done
done