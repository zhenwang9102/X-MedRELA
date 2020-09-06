#!/usr/bin/env bash

cd ./src

bi_rela="treatment"
rand_seed=4219
num_ctx=32
gpu_number=1


CUDA_VISIBLE_DEVICES=$gpu_number python -u main_ration_triple.py \
--target_masking=True \
--binary_rela=$bi_rela \
--graph_file="../data/sub_neighbors_dict_ppmi_perBin_1.pkl" \
--rela_tp_file="../data/final_relation_triples_10rela.pkl" \
--bi_rela_file="../data/${bi_rela}_dataset_neg_1.pkl" \
--rela_list_file="../data/umls_relations_10rela.txt" \
--random_seed=$rand_seed \
--num_contexts=$num_ctx \
--batch_size=128 \
--cooc_batch_size=256 \
--cooc_num_workers=0 \
--rela_batch_size=256 \
--rela_num_workers=0 \
--num_neg_sample=100 \
--learning_rate=0.001 \
--cut_rela_score=True \
--save_dir="./saved_models/owa_${bi_rela}_10_rela_ctx${num_ctx}_rand${rand_seed}_filter"

