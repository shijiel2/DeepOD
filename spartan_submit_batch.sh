#!/bin/bash

conda_env="deepod"
script_name="run.py"

runtime="60:00:00"
mem="256G"
cpus_per_task="16"
a100_gpus="1"
deeplearn_gpus="1"
partition="deeplearn" # a100 feita100 deeplearn

# Dataset paths
dataset="UCR"
dataset_sub="1"

# General parameters
model="COUTA" # "TimesNet" "COUTA" "DeepSVDDTS"  "TranAD" || too slow "DeepIsolationForestTS" requires 1000+ hours to certify
seed=0
subset_size=-1  # -1 means use all data
save_model="False"
dtw_ar_attack="True" # "True" "False"
dtw_ar_attack_budget="1.0" # 1.0 2.0 3.0 4.0 5.0

# Model parameters
seq_len=10
stride=1
epochs=100
epoch_steps=-1  # -1 means use all data
batch_size=64
lr=0.001

# Smooth parameters
sigma=0.1
window_size=2
smooth_count=500

# Iterate arguments
seq_len_list=("50")
partition_list=("a100" "feita100" "deeplearn") # "a100" "feita100" "deeplearn"
dataset_list=("SMAP" "SMD" "UCR" "MSL" "NIPS_TS_Swan" "NIPS_TS_creditcard" "NIPS_TS_Water") # "SMAP" "SMD" "UCR" "MSL" "NIPS_TS_Swan" "NIPS_TS_creditcard" "NIPS_TS_Water"
model_list=("COUTA") # "TimesNet" "COUTA" "DeepSVDDTS"  "TranAD"
sigma_list=("0.5" "1.0") # "0.1" "0.5" "1.0" "2.0" "5.0"
window_size_list=("4")


for partition in "${partition_list[@]}"; do
    exps_root="exps/${partition}/attacked"
    for dataset in "${dataset_list[@]}"; do
        if [ "$dataset" == "UCR" ]; then
            dataset_sub_list=("1" "2")
        else
            dataset_sub_list=("")
        fi
        for dataset_sub in "${dataset_sub_list[@]}"; do
            if [ "$dataset" == "UCR" ]; then
                data_train="dataset/DCdetector_dataset/${dataset}/${dataset}_${dataset_sub}_train.npy"
                data_test="dataset/DCdetector_dataset/${dataset}/${dataset}_${dataset_sub}_test.npy"
                data_test_label="dataset/DCdetector_dataset/${dataset}/${dataset}_${dataset_sub}_test_label.npy"
            else
                data_train="dataset/DCdetector_dataset/${dataset}/${dataset}_train.npy"
                data_test="dataset/DCdetector_dataset/${dataset}/${dataset}_test.npy"
                data_test_label="dataset/DCdetector_dataset/${dataset}/${dataset}_test_label.npy"
            fi
            for model in "${model_list[@]}"; do
                for seq_len in "${seq_len_list[@]}"; do
                    for sigma in "${sigma_list[@]}"; do
                        for window_size in "${window_size_list[@]}"; do
                            # Generate a random experiment name
                            id=$((RANDOM % 9000000 + 1000000))
                            exp_name="${dataset}_${dataset_sub}_${model}_epochs_${epochs}_seq_len_${seq_len}_batch_size_${batch_size}_sigma_${sigma}_w_${window_size}_smooth_count_${smooth_count}_seed_${seed}_id_${id}"

                            # Construct arguments string
                            arguments="--dtw_ar_attack_budget $dtw_ar_attack_budget --dtw_ar_attack $dtw_ar_attack --save_model $save_model --data_train $data_train --data_test $data_test --data_test_label $data_test_label --exps_root $exps_root --exp_name $exp_name --subset_size $subset_size --model $model --seed $seed --seq_len $seq_len --stride $stride --epochs $epochs --epoch_steps $epoch_steps --batch_size $batch_size --lr $lr --sigma $sigma --window_size $window_size --smooth_count $smooth_count"

                            # Run the script with arguments
                            echo "Running experiment: $exp_name"
                            bash spartan_submit.sh --partition $partition --conda-env $conda_env --script-name $script_name --arguments "$arguments" --runtime $runtime --mem $mem --cpus-per-task $cpus_per_task --a100-gpus $a100_gpus --deeplearn-gpus $deeplearn_gpus --id $id
                        done
                    done
                done
            done
        done
    done
done
