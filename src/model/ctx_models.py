import os
import sys
import numpy as np
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class CtxPredictionLINE2(nn.Module):
    def __init__(self, args):
        super(CtxPredictionLINE2, self).__init__()
        self.args = args
        self.embed_dim = args.node_embed_dim
        self.embed_size = args.node_vocab_size
        self.output_V = args.node_vocab_size

        # self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(args.pre_train_nodes), freeze=False)
        self.embeddings = nn.Embedding(self.embed_size, self.embed_dim)
        self.init_embedding()

        # self.target_out = nn.Linear(self.embed_dim, self.output_V, bias=False)
        # self.target_out.weight = self.node_embedding.weight

        self.context_out = nn.Linear(self.embed_dim, self.output_V, bias=False)
        self.emb_out = nn.LogSoftmax(dim=1)

    def init_embedding(self):
        initrange = 0.5 / self.embed_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def line2_ce_loss(self, logits, labels):
        return -torch.sum(labels * self.emb_out(logits))

    def forward(self, x):
        # inputs: [id, id, id, ...]
        x = self.embeddings(x)  # (K, D)
        # print(x.shape)

        # target_scores = self.target_out(all_embeds)  # ()
        contxt_scores = self.context_out(x)

        return contxt_scores
