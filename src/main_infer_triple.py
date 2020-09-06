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
    print('Model Parameters: ')
    print('# contexts to aggregate: {0}'.format(args.num_contexts))
    print('*****************************')

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # global parameters
    args.term_strings = pickle.load(open('../../SurfCon/global_mapping/term_string_mapping.pkl', 'rb'))
    args.concept_strings = pickle.load(open('../../SurfCon/global_mapping/concept_string_mapping.pkl', 'rb'))
    args.term_concept_dict = pickle.load(open('../../SurfCon/global_mapping/term_concept_mapping.pkl', 'rb'))
    args.concept_term_dict = pickle.load(open('../../SurfCon/global_mapping/concept_term_mapping.pkl', 'rb'))
    args.concept_cui_mapping = pickle.load(open('../../SurfCon/global_mapping/concept_to_CUI_mapping.pkl', 'rb'))
    args.id2stype, args.id2srela = pickle.load(open('../data/SN/st_sr_id2str.pkl', 'rb'))
    args.cui2type = pickle.load(open('../data/SN/all_cui_types_dict.pkl', 'rb'))
    all_rela = [x.strip().split(';')[0] for x in open('../data/umls_relations_2.txt').readlines()] + ['n/a']

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
        pickle.load(open('../data_final/' + args.binary_rela + '_dataset_neg_1.pkl', 'rb'))
    np.random.shuffle(train_data)
    print('Data loaded!')
    print('#cooc nodes: ', len(args.all_iv_terms))
    print('#rela tuples: ', len(rela_tuples))
    print('#Train: {0}, #Dev: {1}, #Test: {2}'.format(len(train_data), len(dev_data), len(test_data)))
    print(train_data[0])

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

    print('Begin digitalizing ...')
    # prepare dataloaders
    dev_dataset = loader.BinaryRelaDataset(dev_data, args.node_to_id)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=True)

    test_dataset = loader.BinaryRelaDataset(test_data, args.node_to_id)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    '''build model'''
    model = triple.TupleRationaleModel(args).to(args.device)
    print(model)
    print([(name, p.numel()) for name, p in model.named_parameters()])
    model.load_state_dict(torch.load(args.stored_model_path, map_location=torch.device(args.device)), strict=True)
    model.eval()
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('Total # parameter: {0} w/ embeddings'.format(pytorch_total_params))
    #
    # pytorch_total_params = sum(p.numel() for name, p in model.named_parameters()
    #                            if p.requires_grad and name.count('embeddings') == 0)
    # print('Total # parameter: {0} w/o embeddings'.format(pytorch_total_params))

    sel_index = [args.node_to_id[x] for x in all_kb_terms]
    args.sel_index = torch.tensor(sel_index, device=args.device)

    criterion = nn.BCEWithLogitsLoss()
    results = train_utils.eval_binary_relation(dev_loader, model, criterion, args)
    print('Dev: Acc {0:.3f}, F1 {1:.3f}, P {2:.3f}, R {3:.3f}'.format(
        results[0], results[1], results[2], results[3]))
    results = train_utils.eval_binary_relation(test_loader, model, criterion, args)
    print('Test: Acc {0:.3f}, F1 {1:.3f}, P {2:.3f}, R {3:.3f}'.format(
        results[0], results[1], results[2], results[3]))

    count = 0
    sum_evidences = 0
    tp_samples = {}
    fp_samples = {}

    while count < args.num_output:
        heads, tails, labels = next(iter(test_loader))
        heads, tails = heads.to(args.device), tails.to(args.device)
        head_ctx, tail_ctx = utils.filter_context(heads, tails, model.ctx_predictor, args.sel_index, args.num_contexts)

        labels = labels.type(torch.FloatTensor).to(args.device)
        logits, (tuple_att_weights, pair_scores) = model(head_ctx, tail_ctx)
        tuple_att_weights = tuple_att_weights.squeeze(-1)

        if args.cuda:
            heads = heads.to('cpu').detach().data.numpy()
            tails = tails.to('cpu').detach().data.numpy()
            head_ctx = head_ctx.to('cpu').detach().data.numpy()
            tail_ctx = tail_ctx.to('cpu').detach().data.numpy()

            logits = logits.to('cpu').detach().data.numpy()
            labels = labels.to('cpu').detach().data.numpy()
            tuple_att_weights = tuple_att_weights.to('cpu').detach().data.numpy()
            pair_scores = pair_scores.to('cpu').detach().data.numpy()
        else:
            heads = heads.to('cpu').detach().data.numpy()
            tails = tails.to('cpu').detach().data.numpy()
            head_ctx = head_ctx.detach().data.numpy()
            tail_ctx = tail_ctx.detach().data.numpy()

            logits = logits.detach().data.numpy()
            labels = labels.detach().data.numpy()
            tuple_att_weights = tuple_att_weights.detach().data.numpy()
            pair_scores = pair_scores.detach().data.numpy()

        preds = np.where(utils.sigmoid(logits) >= 0.5, 1., 0.)
        # print('True Label: ', labels)
        # print('Pred Label: ', preds)

        if labels == 1 and preds == 1:
            pair = (args.id_to_node[heads[0]], args.id_to_node[tails[0]])
            tp_samples.setdefault(pair, {})
            tp_samples[pair]['head'] = args.term_strings[pair[0]]
            tp_samples[pair]['tail'] = args.term_strings[pair[1]]
            tp_samples[pair]['pred_score'] = utils.sigmoid(logits)[0]
            tp_samples[pair]['left_concepts'] = [args.concept_strings[x] for x in args.term_concept_dict[pair[0]]]
            tp_samples[pair]['right_concepts'] = [args.concept_strings[x] for x in args.term_concept_dict[pair[1]]]

            mapper = {'id2node': args.id_to_node,
                      'term2str': args.term_strings,
                      'concept2str': args.concept_strings,
                      'id2rela': args.id_to_rela,
                      'term2concept': args.term_concept_dict,
                      'concept2cui': args.concept_cui_mapping}

            evidence_str, evidence_id = miner.recovering_ranked_triples(head_ctx[0], tail_ctx[0],
                                                                        tuple_att_weights[0], pair_scores[0],
                                                                        args.num_top_tuples, mapper, args.sem_net)
            tp_samples[pair]['evidences_str'] = evidence_str
            tp_samples[pair]['evidences_id'] = evidence_id

        elif labels == 0 and preds == 1:
            pair = (args.id_to_node[heads[0]], args.id_to_node[tails[0]])
            fp_samples.setdefault(pair, {})
            fp_samples[pair]['head'] = args.term_strings[pair[0]]
            fp_samples[pair]['tail'] = args.term_strings[pair[1]]
            fp_samples[pair]['pred_score'] = utils.sigmoid(logits)[0]
            fp_samples[pair]['left_concepts'] = [args.concept_strings[x] for x in args.term_concept_dict[pair[0]]]
            fp_samples[pair]['right_concepts'] = [args.concept_strings[x] for x in args.term_concept_dict[pair[1]]]

            mapper = {'id2node': args.id_to_node,
                      'term2str': args.term_strings,
                      'concept2str': args.concept_strings,
                      'id2rela': args.id_to_rela,
                      'term2concept': args.term_concept_dict,
                      'concept2cui': args.concept_cui_mapping}

            evidence_str, evidence_id = miner.recovering_ranked_triples(head_ctx[0], tail_ctx[0],
                                                                        tuple_att_weights[0], pair_scores[0],
                                                                        args.num_top_tuples, mapper, args.sem_net)
            fp_samples[pair]['evidences_str'] = evidence_str
            fp_samples[pair]['evidences_id'] = evidence_id
        else:
            pass
        count += 1
        # print(count)
    pickle.dump([tp_samples, fp_samples], open('../user_study/pred_evidence_tp_fp_sample.pkl', 'wb'), protocol=-1)
    return


if __name__ == '__main__':
    main()
