#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

results_path=$base_path/../results/model_results.csv
if [ -f "$results_path" ]; then
  rm $results_path
fi

dataset_names=('calsnic' 'proact')
seeds=(0 1 2 3 4)

for dataset_name in "${dataset_names[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Running $dataset_name with seed $seed"
        python3 $base_path/../src/train_models.py --dataset_name "$dataset_name" --seed "$seed"
    done
done