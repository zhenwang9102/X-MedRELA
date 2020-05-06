import os
import sys
import numpy as np
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_


class RelaDistMultModel(nn.Module):
    def __init__(self, args, node_embeddings=None):
        super(RelaDistMultModel, self).__init__()
        self.args = args
        self.node_size = args.node_vocab_size
        self.node_embed_dim = args.node_embed_dim
        self.rela_embed_dim = args.rela_embed_dim
        self.rela_size = args.rela_size

        # self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(args.pre_train_nodes), freeze=False)
        if not node_embeddings:
            self.node_embeddings = nn.Embedding(self.node_size, self.node_embed_dim)
            # self.init_node_embedding()
            xavier_uniform_(self.node_embeddings.weight.data)
        else:
            self.node_embeddings = node_embeddings

        self.rela_embeddings = nn.Embedding(self.rela_size, self.rela_embed_dim)
        # self.init_rela_embeddings()
        xavier_uniform_(self.rela_embeddings.weight.data)

        self.emb_out = nn.LogSoftmax(dim=1)

    # def init_node_embedding(self):
    #     initrange = 0.5 / self.node_embed_dim
    #     self.node_embeddings.weight.data.uniform_(-initrange, initrange)

    # def init_rela_embeddings(self):
    #     initrange = 0.5 / self.rela_embed_dim
    #     self.rela_embeddings.weight.data.uniform_(-initrange, initrange)

    def scoring_func(self, heads, tails, relas):
        scores = heads * relas * tails  # (B, M, D)
        # scores = scores.sum(dim=-1)
        scores = torch.sum(scores, dim=-1)
        return scores

    def forward(self, tuples):
        # inputs: (heads, tails, relas)
        heads = self.node_embeddings(tuples[0])  # (B, M, D)
        tails = self.node_embeddings(tuples[1])  # (B, M, D)
        relas = self.rela_embeddings(tuples[2])  # (B, M, D)
        # print(heads.shape, tails.shape, relas.shape)
        scores = self.scoring_func(heads, tails, relas)

        return scores


class RelaTransEModel(nn.Module):
    def __init__(self, args, node_embeddings=None):
        super(RelaTransEModel, self).__init__()
        self.args = args
        self.node_size = args.node_vocab_size
        self.node_embed_dim = args.node_embed_dim
        self.rela_embed_dim = args.rela_embed_dim
        self.rela_size = args.rela_size

        # self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(args.pre_train_nodes), freeze=False)
        if not node_embeddings:
            self.node_embeddings = nn.Embedding(self.node_size, self.node_embed_dim)
            # self.init_node_embedding()
            xavier_uniform_(self.node_embeddings.weight.data)
        else:
            self.node_embeddings = node_embeddings

        self.rela_embeddings = nn.Embedding(self.rela_size, self.rela_embed_dim)
        # self.init_rela_embeddings()
        xavier_uniform_(self.rela_embeddings.weight.data)

        self.emb_out = nn.LogSoftmax(dim=1)

    # def init_node_embedding(self):
    #     initrange = 0.5 / self.node_embed_dim
    #     self.node_embeddings.weight.data.uniform_(-initrange, initrange)
    #
    # def init_rela_embeddings(self):
    #     initrange = 0.5 / self.rela_embed_dim
    #     self.rela_embeddings.weight.data.uniform_(-initrange, initrange)

    def scoring_func(self, heads, tails, relas):
        if self.args.trans_score == 'l1':
            scores = torch.norm(heads + relas - tails, p=1, dim=2)
        else:
            # 'l2' loss function
            scores = torch.norm(heads + relas - tails, p=2, dim=2)
        # scores = torch.norm(heads, p=2, dim=2) + torch.norm(relas, p=2, dim=2) - torch.norm(tails, p=2, dim=2)
        # scores = - torch.abs(scores)  # (B, M)

        return - scores  # negate the score for softmax loss

    def forward(self, tuples):
        # inputs: (heads, tails, relas)
        if self.args.detach_kb_node:
            heads = self.node_embeddings(tuples[0]).detach()  # (B, M, D)
            tails = self.node_embeddings(tuples[1]).detach()  # (B, M, D)
        else:
            heads = self.node_embeddings(tuples[0])  # (B, M, D)
            tails = self.node_embeddings(tuples[1])  # (B, M, D)

        relas = self.rela_embeddings(tuples[2])  # (B, M, D)
        # print(heads.shape, tails.shape, relas.shape)

        scores = self.scoring_func(heads, tails, relas)
        return scores
