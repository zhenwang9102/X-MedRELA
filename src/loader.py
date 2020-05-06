import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

import utils


class BinaryEvidenceDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, samples, node_to_id, evidences):
        'Initialization'
        self.head_dict = {}
        self.tail_dict = {}
        self.label_dict = {}
        self.evidences = evidences
        for idx, sample in enumerate(samples):
            self.head_dict[idx] = node_to_id[sample[0]]
            self.tail_dict[idx] = node_to_id[sample[1]]
            self.label_dict[idx] = sample[-1]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_dict)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        head = self.head_dict[index]
        tail = self.tail_dict[index]
        label = self.label_dict[index]
        evidences = self.evidences[(head, tail)]
        head_ctx = np.array([x[0] for x in evidences])
        tail_ctx = np.array([x[1] for x in evidences])
        rela_ctx = np.array([x[2] for x in evidences])

        return head, tail, label, head_ctx, tail_ctx, rela_ctx


class BinaryRelaDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, samples, node_to_id):
        'Initialization'
        self.head_dict = {}
        self.tail_dict = {}
        self.label_dict = {}
        for idx, sample in enumerate(samples):
            self.head_dict[idx] = node_to_id[sample[0]]
            self.tail_dict[idx] = node_to_id[sample[1]]
            self.label_dict[idx] = sample[-1]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_dict)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        head = self.head_dict[index]
        tail = self.tail_dict[index]
        label = self.label_dict[index]

        return head, tail, label


class RelationDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, all_tuples, node_to_id, rela_to_id, num_neg_samples, all_entity_ids):
        'Initialization'
        self.num_neg_samples = num_neg_samples
        self.all_entity_ids = set(all_entity_ids)
        self.heads_dict = {}
        self.tails_dict = {}
        self.relas_dict = {}
        for idx, tp in enumerate(all_tuples):
            self.heads_dict[idx] = node_to_id[tp[0]]
            self.tails_dict[idx] = node_to_id[tp[1]]
            self.relas_dict[idx] = rela_to_id[tp[2]]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.heads_dict)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        pos_head = self.heads_dict[index]
        pos_tail = self.tails_dict[index]
        pos_rela = self.relas_dict[index]

        if np.random.random() < 0.5:
            false_heads = np.random.choice(list(self.all_entity_ids - {pos_head}), self.num_neg_samples)
            label = np.random.randint(self.num_neg_samples)
            heads = np.insert(false_heads, label, pos_head)
            tails = np.array([pos_tail] * (self.num_neg_samples + 1))
            relas = np.array([pos_rela] * (self.num_neg_samples + 1))
        else:
            false_tails = np.random.choice(list(self.all_entity_ids - {pos_tail}), self.num_neg_samples)
            label = np.random.randint(self.num_neg_samples)
            tails = np.insert(false_tails, label, pos_tail)
            heads = np.array([pos_head] * (self.num_neg_samples + 1))
            relas = np.array([pos_rela] * (self.num_neg_samples + 1))

        return heads, tails, relas, label


class CoocGraphDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, neighbor_dict, node_to_id):
        'Initialization'
        self.node_dict = {}
        self.label_dict = {}
        for idx, node in enumerate(neighbor_dict.keys()):
            self.node_dict[idx] = node_to_id[node]
            label_vec = np.zeros(len(node_to_id))
            for y in neighbor_dict[node]:
                if y[0] in node_to_id:
                    label_vec[node_to_id[y[0]]] = y[1]
            self.label_dict[idx] = label_vec / float(np.sum(label_vec))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.node_dict)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        X = self.node_dict[index]
        y = self.label_dict[index]

        return X, y


def graph_batching(batch_data):
    x_batch = []
    y_batch = []
    for data in batch_data:
        x_batch.append(data['x'])
        y_batch.append(data['y'])

    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch


def make_idx_graph(graph_data, node_to_id):
    data_dicts = []
    for node in graph_data.keys():
        cur_dict = {}
        cur_dict['x'] = node_to_id[node]

        label_vec = np.zeros(len(node_to_id))
        for y in graph_data[node]:
            if y[0] in node_to_id:
                label_vec[node_to_id[y[0]]] = y[1]
        cur_dict['y'] = label_vec / float(np.sum(label_vec))
        data_dicts.append(cur_dict)

    return data_dicts


def make_idx_relation(dataset, args):
    data_dicts = []
    for data in dataset:
        # data: a tuple - (t1, t2, rela)
        cur_dict = {}
        cur_dict['t1'] = data[0]
        cur_dict['t2'] = data[1]
        cur_dict['t1_id'] = args.node_to_id[data[0]]
        cur_dict['t2_id'] = args.node_to_id[data[1]]
        cur_dict['label'] = data[-1]

        data_dicts.append(cur_dict)
    return data_dicts


def batch_process_relation(batch_data, args, Train=None):
    t1_batch = []
    t2_batch = []
    label_batch = []
    t1_contexts = []
    t2_contexts = []
    aux_label_batch = []

    for data in batch_data:
        t1_batch.append(data['t1_id'])
        t2_batch.append(data['t2_id'])
        label_batch.append(data['label'])

        # sample neighbors
        t1_ctx = context_retriever(data['t1'], args.cos_neighbors, args.num_contexts, args.node_to_id)
        t1_contexts.append(t1_ctx)

        t2_ctx = context_retriever(data['t2'], args.cos_neighbors, args.num_contexts, args.node_to_id)
        t2_contexts.append(t2_ctx)

    t1_batch = np.array(t1_batch)
    t2_batch = np.array(t2_batch)
    t1_contexts = utils.pad_sequence(t1_contexts, padder=0)
    t2_contexts = utils.pad_sequence(t2_contexts, padder=0)
    label_batch = np.array(label_batch)

    if args.use_context and args.use_aux_loss:
        for i in range(len(batch_data)):
            cur_aux_label = []
            for t1 in t1_contexts[i]:
                for t2 in t2_contexts[i]:
                    if (int(t1), int(t2)) in args.link_set:
                        cur_aux_label.append(1)
                    else:
                        cur_aux_label.append(0)
            aux_label_batch.append(cur_aux_label)
        aux_label_batch = np.array(aux_label_batch)

    return [t1_batch, t1_contexts, t2_batch, t2_contexts], label_batch, aux_label_batch


def context_retriever(term_id, neighbor_dict, num_contexts, node_to_id):
    # neighbor_dict: return a list of contexts (w/ term id)
    # node_to_id: convert term id to embed id
    t1_neighbors = [x[0] for x in neighbor_dict[term_id]]
    t1_ctx = t1_neighbors[:num_contexts]

    if term_id in t1_ctx:
        t1_ctx.remove(term_id)

    t1_ctx = [node_to_id[x] for x in t1_ctx]
    return t1_ctx


def node_mapping(node_embed_path, node_to_id=None):
    f = open(node_embed_path).readlines()
    # print('Node embeddings: ', f[0].strip())
    node_dict = {int(x.strip().split()[0]): np.array(x.strip().split()[1::], dtype=np.float32)
                 for x in f[1::]}

    # print(len(node_dict))
    if not node_to_id:
        terms_mat = np.zeros((len(node_dict) + 1, int(f[0].strip().split()[1])))
        terms_to_idx = {-1: 0}
        idx_to_terms = {0: -1}
        for idx, term_id in enumerate(node_dict.keys()):
            terms_mat[idx + 1, :] = node_dict[term_id].reshape(1, -1)
            terms_to_idx[term_id] = idx + 1
            idx_to_terms[idx + 1] = term_id

        return terms_to_idx, idx_to_terms, terms_mat
    else:
        terms_mat = np.zeros((len(node_to_id), int(f[0].strip().split()[1])))
        for term_id, idx in node_to_id.items():
            terms_mat[idx, :] = node_dict[term_id].reshape(-1)

        return terms_mat
