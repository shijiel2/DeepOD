#!/bin/bash

conda_env="deepod"
script_name="run.py"

# Dataset paths
dataset="SMAP" # SMAP SMD MSL NIPS_TS_Swan NIPS_TS_creditcard NIPS_TS_Water

data_train="dataset/DCdetector_dataset/${dataset}/${dataset}_train.npy"
data_test="dataset/DCdetector_dataset/${dataset}/${dataset}_test.npy"
data_test_label="dataset/DCdetector_dataset/${dataset}/${dataset}_test_label.npy"


save_model="True"
load_model="None"
load_noise_scores="None"
dtw_ar_attack="True"


# General parameters
exps_root="exps"
model="COUTA"
seed=0
subset_size=-1  # -1 means use all data

# Model parameters
seq_len=50
stride=1
epochs=20
epoch_steps=-1  # -1 means use all data
batch_size=64
lr=0.001

# Smooth parameters
sigma=0.5
window_size=4
smooth_count=500

# Generate a random experiment name
id=$((RANDOM % 9000000 + 1000000))
exp_name="${dataset}_${model}_epochs_${epochs}_seq_len_${seq_len}_batch_size_${batch_size}_sigma_${sigma}_w_${window_size}_smooth_count_${smooth_count}_seed_${seed}_id_${id}"

# Construct arguments string
arguments="--data_train $data_train \
          --data_test $data_test \
          --data_test_label $data_test_label \
          --exps_root $exps_root \
          --exp_name $exp_name \
          --subset_size $subset_size \
          --model $model \
          --seed $seed \
          --seq_len $seq_len \
          --stride $stride \
          --epochs $epochs \
          --epoch_steps $epoch_steps \
          --batch_size $batch_size \
          --lr $lr \
          --sigma $sigma \
          --window_size $window_size \
          --smooth_count $smooth_count \
          --load_model $load_model \
          --load_noise_scores $load_noise_scores\
          --save_model $save_model\
          --dtw_ar_attack $dtw_ar_attack"

# Run the script with arguments
echo "Running experiment: $exp_name"
# python -m debugpy --listen 5678 --wait-for-client $script_name $arguments
python $script_name $arguments
