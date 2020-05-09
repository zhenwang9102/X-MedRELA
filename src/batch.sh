#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=2 python -u main_ration_tuple.py \
#--kb_model='transE' \
#--trans_score='l1' \
#--random_seed=51 \
#--num_contexts=32 \
#--batch_size=64 \
#--cooc_batch_size=256 \
#--cooc_num_workers=0 \
#--rela_batch_size=256 \
#--rela_num_workers=0 \
#--num_neg_sample=100 \
#--learning_rate=0.001 \
#--add_hloss=False \
#--detach_kb_node=False \
#--save_dir='./saved_models_2/tuple_transE_l1_32_relu_att' \


bi_rela="symptom"
rand_seed=4219
num_ctx=32
gpu_number=1

#echo $bi_rela
#echo $rand_seed
#echo $num_ctx

#for rand_seed in 4219 7321 8648 42 1716
#do
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
#done
#treatment
#causality
#contraindication
#prevention
#symptom


#cooccurrence



# Experiments for fb15k relations
#CUDA_VISIBLE_DEVICES=2 python -u main_ration_triple_2.py \
#--binary_rela='treatment' \
#--bi_rela_file='../final_data/fb15k_bi_nominated_for_dataset_neg_1.pkl' \
#--graph_file='../final_data/fb15k_train_neighbors_dict.pkl' \
#--rela_tp_file='../data/fb15k_relation_triples.pkl' \
#--rela_list_file='../data/fb15k_relations.txt' \
#--kb_model='transE' \
#--trans_score='l1' \
#--random_seed=41 \
#--num_contexts=16 \
#--batch_size=8 \
#--cooc_batch_size=256 \
#--cooc_num_workers=0 \
#--rela_batch_size=256 \
#--rela_num_workers=0 \
#--num_neg_sample=100 \
#--learning_rate=0.001 \
#--cut_rela_score=True \
#--save_dir='./saved_models/fb15k_nominated_for_all_rela' \



#CUDA_VISIBLE_DEVICES=2 python -u main_ration_path.py \
#--path_rnn='lstm' \
#--kb_model='transE' \
#--trans_score='l1' \
#--num_contexts=32 \
#--batch_size=40 \
#--cooc_batch_size=256 \
#--cooc_num_workers=0 \
#--rela_batch_size=256 \
#--rela_num_workers=0 \
#--num_neg_sample=100 \
#--learning_rate=0.001 \
#--save_dir='./saved_models/path_transE_l1_lstm_noheadtail_32_newatt' \


#CUDA_VISIBLE_DEVICES=1 python -u main_ration_2.py \
#--batch_size=23 \
#--cooc_batch_size=256 \
#--rela_batch_size=256 \
#--num_neg_sample=100 \
#--save_dir='./saved_models/tuple_model_noheadtail' \



#CUDA_VISIBLE_DEVICES=2 python -u main_ration_infer.py \
#--stored_model_path='./saved_models/tuple_transE_noheadtail/best_epoch_0.pt' \
