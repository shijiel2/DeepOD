#!/bin/bash

conda_env="deepod"
script_name="run.py"

# Dataset paths
dataset="SMAP"
dataset_sub=""

data_train="dataset/DCdetector_dataset/${dataset}/${dataset}_train.npy"
data_test="dataset/DCdetector_dataset/${dataset}/${dataset}_test.npy"
data_test_label="dataset/DCdetector_dataset/${dataset}/${dataset}_test_label.npy"


load_model="exps/SMAP_TimesNet_epochs_10_seq_len_10_batch_size_64_sigma_0.1_w_2_smooth_count_500_seed_0_id_1004868/model.pkl"
load_noise_scores="exps/SMAP_TimesNet_epochs_10_seq_len_10_batch_size_64_sigma_0.1_w_2_smooth_count_500_seed_0_id_1004868/saved_noise_scores.pkl"


# General parameters
exps_root="exps"
model="TimesNet"
seed=0
subset_size=10000  # -1 means use all data

# Model parameters
seq_len=10
stride=1
epochs=10
epoch_steps=-1  # -1 means use all data
batch_size=64
lr=0.001

# Smooth parameters
sigma=0.1
window_size=2
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
          --load_noise_scores $load_noise_scores"

# Run the script with arguments
echo "Running experiment: $exp_name"
python $script_name $arguments
