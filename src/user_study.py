import os
import sys
import time
import pickle
import argparse
import numpy as np
import networkx as nx
from datetime import datetime
import sklearn.metrics as metrics
# from torchnlp.word_to_vector import CharNGram

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import utils
import loader
import train_utils
from argparser import *
import recover_evidences as miner
import model.triple_models as triple


def main():
    args = Config()

    print('********Key parameters:******')
    print('Use GPU? {0}'.format(torch.cuda.is_available()))
    # print('Model Parameters: ')
    # print('# contexts to aggregate: {0}'.format(args.num_contexts))
    print('*****************************')

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # path to store
    # args.save_dir = './saved_models/realtion_model_per{0}_{1}'.format(args.per, args.days)
    # print(args.save_dir)

    # global parameters
    args.term_strings = pickle.load(open('../../SurfCon/global_mapping/term_string_mapping.pkl', 'rb'))
    args.concept_strings = pickle.load(open('../../SurfCon/global_mapping/concept_string_mapping.pkl', 'rb'))
    args.term_concept_dict = pickle.load(open('../../SurfCon/global_mapping/term_concept_mapping.pkl', 'rb'))
    args.concept_term_dict = pickle.load(open('../../SurfCon/global_mapping/concept_term_mapping.pkl', 'rb'))
    args.concept_cui_mapping = pickle.load(open('../../SurfCon/global_mapping/concept_to_CUI_mapping.pkl', 'rb'))
    args.id2stype, args.id2srela = pickle.load(open('../data/SN/st_sr_id2str.pkl', 'rb'))
    # args.sem_net = nx.read_gpickle('../data/SN/sn_str_graph.gpickle')  #
    # args.sem_net = pickle.load(open('../data/SN/emp_type_triples.pkl', 'rb'))
    args.cui2type = pickle.load(open('../data/SN/all_cui_types_dict.pkl', 'rb'))
    # args.cos_neighbors = pickle.load(open('../data/all_iv_terms_cos_knn_200.pkl', 'rb'))
    # all_rela = [x.strip().split(';')[0] for x in open('../data/umls_relations.txt').readlines()]
    all_rela = [x.strip().split(';')[0] for x in open('../data/umls_relations_3.txt').readlines()] + ['n/a']

    '''load all three types of data'''
    cooc_graph = pickle.load(open('../data/sub_neighbors_dict_ppmi_perBin_1.pkl', 'rb'))
    args.all_iv_terms = list(cooc_graph.keys())
    # rela_tuples = pickle.load(open('../data/final_relation_tuples.pkl', 'rb'))
    rela_tuples = pickle.load(open('../data/final_relation_tuples_2.pkl', 'rb'))
    all_kb_terms = list(set([x[0] for x in rela_tuples] + [x[1] for x in rela_tuples]))

    meta_triples = []
    for tp in rela_tuples:
        t1_concepts = args.term_concept_dict[tp[0]]
        t2_concepts = args.term_concept_dict[tp[1]]
        # print(t1_concepts, t2_concepts)
        for t1c in t1_concepts:
            for t2c in t2_concepts:
                if t1c in args.concept_term_dict and t2c in args.concept_term_dict:
                    t1cui = args.concept_cui_mapping[t1c]
                    t2cui = args.concept_cui_mapping[t2c]
                    if t1cui in args.cui2type and t2cui in args.cui2type:
                        for c1type in args.cui2type[t1cui]:
                            for c2type in args.cui2type[t2cui]:
                                meta_triples.append((c1type, c2type, tp[-1]))

    print(len(meta_triples))
    args.sem_net = list(set(meta_triples))

    # train_data, dev_data, test_data = pickle.load(open('../data/treatment_dataset_neg_1.pkl', 'rb'))
    train_data, dev_data, test_data = \
        pickle.load(open('../final_data/' + args.binary_rela + '_dataset_neg_1.pkl', 'rb'))
    np.random.shuffle(train_data)
    print('Data loaded!')
    print('#cooc nodes: ', len(args.all_iv_terms))
    print('#rela tuples: ', len(rela_tuples))
    print('#Train: {0}, #Dev: {1}, #Test: {2}'.format(len(train_data), len(dev_data), len(test_data)))
    print(train_data[0])

    # testing
    # train_pos = train_pos[:100]
    # dev = dev[:10]
    # iv_test = iv_test[:10]

    '''data pre-processing'''
    # node mapping
    all_terms = list(cooc_graph.keys())
    args.node_to_id = {node: idx for idx, node in enumerate(all_terms)}
    args.id_to_node = {idx: node for node, idx in args.node_to_id.items()}
    args.node_vocab_size = len(args.node_to_id)
    args.node_embed_dim = args.node_embed_dim
    all_term_ids = [args.node_to_id[x] for x in all_terms]

    # rela mapping
    print('# RELA: ', len(all_rela))
    args.rela_to_id = {rela: idx for idx, rela in enumerate(all_rela)}
    args.id_to_rela = {idx: rela for rela, idx in args.rela_to_id.items()}
    args.rela_size = len(args.rela_to_id)
    args.rela_embed_dim = args.rela_embed_dim

    '''Begin user study ...'''
    pred_tp, pred_fp = pickle.load(open('../user_study/pred_evidence_tp_fp_sample.pkl', 'rb'))
    exist_tp, exist_fp = pickle.load(open('../user_study/exist_evidence_tp_fp_sample.pkl', 'rb'))

    # f = open('../user_study/fp_evidence_examples.txt', 'w')
    count = 0
    for i in range(len(test_data)):
        test_case = test_data[i]
        pair = (test_case[0], test_case[1])
        # label = test_case[2]
        if pair in pred_tp and pair in exist_tp:
            count += 1
            print('----------------------------------Case study {0}---------------------------------\n'.format(count))
            print('All models predict the **may_treat** relation between t1 term '
                  '<span style="color:red">{1} {2}</span> and t2 term <span style="color:blue">{3} {4}</span> '
                  'with the following rationales.\n'.format(
                   0, args.term_strings[pair[0]], pred_tp[pair]['left_concepts'],
                   args.term_strings[pair[1]], pred_tp[pair]['right_concepts']))
            print('Please answer the following questions:\\')

            print('1 Are you familiar with t1 and t2 terms? (Yes/No/Kind of)\\')
            print('-> \\')
            print('2 Check each rationale and answer this question: '
                  'Is which degree is rationale helpful for you to trust the prediction?\\')
            print('(0\~3, 0: no helpful; 1: a little bit helpful; 2: helpful; 3: very helpful)\n')
            # print('Prediction Score: ', round(pred_tp[pair]['pred_score'], 5))
            # print('t1 term {0}: {1}'.format(pair[0], args.term_strings[pair[0]]))
            # print(pred_tp[pair]['left_concepts'])
            # print('t2 term {0}: {1}'.format(pair[1], args.term_strings[pair[1]]))
            # print(pred_tp[pair]['right_concepts'])

            print('\nModel 1\'s Rationale Set: \\')
            ii = 0
            for evid in pred_tp[pair]['evidences_id']:
                if evid[0] == pair[0] and evid[1] == pair[1]:
                    pass
                else:
                    ii += 1
                    print('<span style="color:red">{0}</span> &rarr; {1} &rarr; '
                          '<span style="color:blue">{2}</span> [helpful? (0\~3)]->\\'.format(
                           args.term_strings[evid[0]],
                           args.id_to_rela[evid[2]],
                           args.term_strings[evid[1]]))
                    if ii > 5:
                        break

            print('Model 2\'s Evidence Set\\')
            for evid in exist_tp[pair]['evidences_id'][:5]:
                print('<span style="color:red">{0}</span> &rarr; {1} &rarr; '
                      '<span style="color:blue">{2}</span> [helpful? (0\~3)]->\\'.format(
                       args.term_strings[evid[0]],
                       args.id_to_rela[evid[2]],
                       args.term_strings[evid[1]]))

            print('3 Please rank all sets of rationales based on overall how much they '
                  'help you trust the model prediction (e.g., 1 > 2). Note that it is ok to reject them if '
                  'both models are unhelpful as 1 == 2 = 0: \\')
            print('-> \\')
            print('-----------------------------------End-----------------------------------\n\n\n')
            if count > 50:
                break

    return


if __name__ == '__main__':
    main()
