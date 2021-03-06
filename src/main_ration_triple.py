import os
import sys
import time
import pickle
import argparse
import numpy as np
import networkx as nx
from datetime import datetime
from collections import Counter
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import loader
import utils
import train_utils
import model.triple_models as triple
from argparser import *


def main():
    args = Config()

    print('********Key parameters:******')
    print('Use GPU? {0}'.format(torch.cuda.is_available()))
    print(args.random_seed)
    
    print(args.save_dir)
    print(args.graph_file)
    print(args.rela_tp_file)
    print(args.bi_rela_file)
    print(args.rela_list_file)

    print('Model Parameters: ')
    print('Current RELA: {0}'.format(args.binary_rela))
    print('# contexts to aggregate: {0}'.format(args.num_contexts))
    print('Cut relation score? {0}'.format(args.cut_rela_score))
    print('Cut triple score? {0}'.format(args.cut_triple_score))
    if args.cut_triple_score:
        print('threshold for triple scores: {0}'.format(args.triple_score))
    print('*****************************')

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # path to store
    # args.save_dir = './saved_models/realtion_model_per{0}_{1}'.format(args.per, args.days)
    # print(args.save_dir)

    # global parameters
    # args.term_strings = pickle.load(open('../../global_mapping/term_string_mapping.pkl', 'rb'))
    # args.term_concept_dict = pickle.load(open('../../global_mapping/term_concept_mapping.pkl', 'rb'))
    # args.concept_term_dict = pickle.load(open('../../global_mapping/concept_term_mapping.pkl', 'rb'))
    # args.cos_neighbors = pickle.load(open('../data/all_iv_terms_cos_knn_200.pkl', 'rb'))

    '''load all three types of data'''
    # cooc_graph = pickle.load(open('../data/sub_neighbors_dict_ppmi_perBin_1.pkl', 'rb'))
    cooc_graph = pickle.load(open(args.graph_file, 'rb'))
    args.all_iv_terms = list(cooc_graph.keys())

    # rela_tuples = pickle.load(open('../data_final/final_relation_triples_5rela.pkl', 'rb'))
    rela_tuples = pickle.load(open(args.rela_tp_file, 'rb'))
    all_kb_terms = list(set([x[0] for x in rela_tuples] + [x[1] for x in rela_tuples]))
    # top_kb_terms = [k for k, c in Counter(all_kb_terms).items() if c >= 10]

    # train_data, dev_data, test_data = pickle.load(open('../data/treatment_dataset_neg_1.pkl', 'rb'))
    # train_data, dev_data, test_data = \
    #     pickle.load(open('../data_final/' + args.binary_rela + '_dataset_neg_1.pkl', 'rb'))
    train_data, dev_data, test_data = pickle.load(open(args.bi_rela_file, 'rb'))

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
    print(list(args.node_to_id.items())[:10])

    # rela mapping
    # all_rela = [x.strip().split(';')[0] for x in open('../data/umls_relations_final.txt').readlines()] + ['n/a']
    all_rela = [x.strip().split(';')[0] for x in open(args.rela_list_file).readlines()] + ['n/a']
    print(all_rela)
    print('# RELA: ', len(all_rela))
    args.rela_to_id = {rela: idx for idx, rela in enumerate(all_rela)}
    args.id_to_rela = {idx: rela for rela, idx in args.rela_to_id.items()}
    args.rela_size = len(args.rela_to_id)
    args.rela_embed_dim = args.rela_embed_dim
    print(args.rela_to_id)

    rela_map = {'treatment': ['may_treat', 'may_by_treated_by'],
                'causality': ['causes', 'caused_by'],
                'contraindication': ['contraindicates', 'contraindicated_by'],
                'prevention': ['may_prevent', 'may_be_prevented_by'],
                'symptom': ['symptom_of', 'has_symptom']}
    args.forw_rela = args.rela_to_id[rela_map[args.binary_rela][0]]
    args.back_rela = args.rela_to_id[rela_map[args.binary_rela][1]]

    print('Begin digitalizing ...')
    # prepare dataloaders
    args.cooc_batch_size = 2**(int(np.log2(len(args.all_iv_terms) // (len(train_data) // args.batch_size))))
    # print(args.cooc_batch_size)
    args.rela_batch_size = 2**(int(np.log2(len(rela_tuples) // (len(train_data) // args.batch_size))))
    # print(args.rela_batch_size)

    # prepare dataloaders
    graphdataset = loader.CoocGraphDataset(cooc_graph, args.node_to_id)
    graph_loader = DataLoader(graphdataset, batch_size=args.cooc_batch_size, shuffle=True,
                              num_workers=args.cooc_num_workers)

    relationdataset = loader.RelationDataset(rela_tuples,
                                             args.node_to_id,
                                             args.rela_to_id,
                                             args.num_neg_sample,
                                             all_term_ids)
    relation_loader = DataLoader(relationdataset, batch_size=args.rela_batch_size, shuffle=True,
                                 num_workers=args.rela_num_workers)

    train_dataset = loader.BinaryRelaDataset(train_data, args.node_to_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    dev_dataset = loader.BinaryRelaDataset(dev_data, args.node_to_id)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size // 2, shuffle=False)

    test_dataset = loader.BinaryRelaDataset(test_data, args.node_to_id)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size // 2, shuffle=False)

    '''build model'''
    model = triple.TupleRationaleModel(args).to(args.device)
    print(model)
    print([(name, p.numel()) for name, p in model.named_parameters()])
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('Total # parameter: {0} w/ embeddings'.format(pytorch_total_params))
    #
    # pytorch_total_params = sum(p.numel() for name, p in model.named_parameters()
    #                            if p.requires_grad and name.count('embeddings') == 0)
    # print('Total # parameter: {0} w/o embeddings'.format(pytorch_total_params))

    att_criterion = utils.HLoss()
    rela_criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer_ctx = optim.Adam(model.ctx_predictor.parameters(), lr=args.learning_rate)
    # optimizer_rela = optim.Adam(model.rela_model.parameters(), lr=args.learning_rate)

    last_epoch = 0
    best_on_dev = 0.0
    train_loss = []
    train_logits = []
    train_labels = []
    best_res_dev = []
    best_res_test = []

    sel_index = [args.node_to_id[x] for x in all_kb_terms]
    args.sel_index = torch.tensor(sel_index, device=args.device)

    args.log_interval = (len(train_data) // args.batch_size) // 2
    # num_batches = len(train_idx_data) // args.batch_size
    print('Begin training...')
    for epoch in range(args.num_epochs):
        print(datetime.now().strftime("%m/%d/%Y %X"))
        model.train()
        steps = 0
        '''
        np.random.shuffle(train_idx_data)
        for i in range(num_batches):
            train_batch = train_idx_data[i * args.batch_size: (i + 1) * args.batch_size]
            if i == num_batches - 1:
                train_batch = train_idx_data[i * args.batch_size::]

            list_xs, labels, aux_labels = loader.batch_process_relation(train_batch, args, Train=True)

            labels = torch.tensor(labels, dtype=torch.float, device=args.device)  # (B * L)
            aux_labels = torch.tensor(aux_labels, dtype=torch.float, device=args.device)  # (B * L)
            list_xs = [torch.tensor(x, device=args.device) for x in list_xs]
        '''
        start = time.time()
        for heads, tails, labels in train_loader:

            # co-oc graph
            optimizer.zero_grad()
            cooc_x, cooc_y = next(iter(graph_loader))
            # print('1', time.time() - start)
            cooc_x, cooc_y = cooc_x.to(args.device), cooc_y.type(torch.FloatTensor).to(args.device)
            cocc_logits = model.ctx_predictor(cooc_x)
            ctx_loss = model.ctx_predictor.line2_ce_loss(cocc_logits, cooc_y)
            # print(ctx_loss)
            ctx_loss.backward()
            nn.utils.clip_grad_norm_(model.ctx_predictor.parameters(), args.clip_grad)
            optimizer.step()
            # print('2', time.time() - start)

            # kb tuples
            optimizer.zero_grad()
            kb_heads, kb_tails, kb_relas, kb_labels = next(iter(relation_loader))
            # print('3', time.time() - start)
            kb_heads, kb_tails, kb_relas = kb_heads.to(args.device), kb_tails.to(args.device), kb_relas.to(args.device)
            kb_labels = kb_labels.type(torch.LongTensor).to(args.device)
            kb_logits = model.rela_model([kb_heads, kb_tails, kb_relas])
            rela_loss = rela_criterion(kb_logits, kb_labels)
            # print(rela_loss)
            rela_loss.backward()
            nn.utils.clip_grad_norm_(model.rela_model.parameters(), args.clip_grad)
            optimizer.step()
            # print('4', time.time() - start)

            # filter contexts
            heads, tails = heads.to(args.device), tails.to(args.device)
            # forw_rela, back_rela = torch.LongTensor(forw_rela).to(args.device), torch.LongTensor(back_rela).to(args.device)
            head_ctx, tail_ctx =\
                utils.filter_context(heads, tails, model.ctx_predictor, args.sel_index, args.num_contexts)

            target_masks = utils.filter_target_pairs(heads, head_ctx, tails, tail_ctx, args.forw_rela, args.back_rela, args)
            target_masks = torch.tensor(target_masks, dtype=torch.bool).to(args.device)
            # binary relation prediction
            labels = labels.type(torch.FloatTensor).to(args.device)
            optimizer.zero_grad()
            logits, (tuple_att_weights, norm_scores) = model(head_ctx, tail_ctx, target_masks)
            loss = criterion(logits, labels)
            # print(loss)
            train_loss.append(loss.item())

            if args.add_hloss:
                Hloss = att_criterion(tuple_att_weights)
                # print(Hloss)
                loss += Hloss * 0.001

            # print(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            if args.cuda:
                logits = logits.to('cpu').detach().data.numpy()
                labels = labels.to('cpu').detach().data.numpy()
            else:
                logits = logits.detach().data.numpy()
                labels = labels.detach().data.numpy()

            # print(logits.shape)
            train_logits.append(logits)
            train_labels.append(labels)

            # evaluation
            steps += 1
            if steps % args.log_interval == 0:
                train_golds = np.concatenate(train_labels)
                train_logits = np.concatenate(train_logits)
                train_preds = np.where(utils.sigmoid(train_logits) >= 0.5, 1, 0)
                # train_preds = np.argmax(train_logits)
                # print(train_golds.shape, train_preds.shape)
                train_acc = metrics.accuracy_score(train_golds, train_preds)
                train_f1 = metrics.f1_score(train_golds, train_preds)
                # train_acc = np.sum(np.equal(train_preds, train_golds)) / train_logits.shape[0]
                print("Epoch-{0}, steps-{1}: Train Loss - {2:.5}, Train ACC - {3:.5}, Train F1 - {4:.5}".
                      format(epoch, steps, np.mean(train_loss), train_acc, train_f1))

                # results = train_utils.eval_binary_relation(dev_loader, model, criterion, args)
                # print("Epoch-{0}: Dev ACC: {1:.5} F1: {2:.5}".format(epoch, results[0], results[1]))
                #
                # results = train_utils.eval_binary_relation(test_loader, model, criterion, args)
                # print("--- Testing: Test ACC: {1:.5} F1: {2:.5}".format(epoch, results[0], results[1]))

                train_loss = []
                train_logits = []
                train_labels = []

        if epoch % args.test_interval == 0:
            dev_results = train_utils.eval_binary_relation(dev_loader, model, criterion, args)
            print("Epoch-{0}: Dev ACC: {1:.5} P: {2:.5} R: {3:.5} F1: {4:.5}".format(
                epoch, dev_results[0], dev_results[2], dev_results[3], dev_results[1]))
            # print('Step - ', datetime.now().strftime("%m/%d/%Y %X"))

            if dev_results[1] > best_on_dev:  # macro f1 score
                print(datetime.now().strftime("%m/%d/%Y %X"))
                best_on_dev = dev_results[1]
                best_res_dev = dev_results
                last_epoch = epoch

                test_results = train_utils.eval_binary_relation(test_loader, model, criterion, args)
                print("--- Testing: Test ACC: {1:.5} P: {2:.5} R: {3:.5} F1: {4:.5}".format(
                    epoch, test_results[0], test_results[2], test_results[3], test_results[1]))
                best_res_test = test_results

                if args.save_best:
                    utils.save(model, args.save_dir, 'best', epoch)

            else:
                if epoch - last_epoch > args.early_stop_epochs and epoch > args.min_epochs:
                    print('Binary relation: {0}'.format(args.binary_rela))
                    print('Best Performance at epoch: {0}'.format(last_epoch))
                    print('Best Dev: A {0:.3f}\t P {1:.3f}\t R {2:.3f}\t F {3:.3f}'.format(best_res_dev[0],
                                                                                           best_res_dev[2],
                                                                                           best_res_dev[3],
                                                                                           best_res_dev[1]))

                    print('Best Test: A {0:.3f}\t P {1:.3f}\t R {2:.3f}\t F {3:.3f}'.format(best_res_test[0],
                                                                                            best_res_test[2],
                                                                                            best_res_test[3],
                                                                                            best_res_test[1]))

                    print('Early stop at {0} epoch.'.format(epoch))
                    break

        # if epoch % args.save_interval == 0:
        #     utils.save(model, args.save_dir, 'snapshot', epoch)

    return


if __name__ == '__main__':
    main()
