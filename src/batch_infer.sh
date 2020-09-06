#!/usr/bin/env bash

cd ./src

CUDA_VISIBLE_DEVICES=2 python -u main_infer_triple.py \
--binary_rela='treatment' \
--kb_model='transE' \
--trans_score='l1' \
--random_seed=41 \
--num_contexts=32 \
--num_output=5766 \
--num_top_tuples=10 \
--num_top_rela=5 \
--filter_type=True \
--stored_model_path='./saved_models/owa_treatment_new_rela_ctx32_rand42/best_epoch_91.pt' \
--stored_preds_path='../user_study/pred_evidence_tp_fp_sample.pkl'


python -u user_study.py \
--stored_preds_path='../user_study/pred_evidence_tp_fp_sample.pkl'

