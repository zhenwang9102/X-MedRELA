import os
import sys
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ctx_models import *
from model.rela_models import *


class TupleRationaleModel(nn.Module):
    def __init__(self, args):
        super(TupleRationaleModel, self).__init__()
        self.args = args
        self.num_contexts = args.num_contexts
        self.embed_dim = args.node_embed_dim
        self.embed_size = args.node_vocab_size
        self.output_V = args.node_vocab_size

        self.ctx_predictor = CtxPredictionLINE2(args)

        if self.args.kb_model == 'transE':
            self.rela_model = RelaTransEModel(args, self.ctx_predictor.embeddings)
        else:
            self.rela_model = RelaDistMultModel(args, self.ctx_predictor.embeddings)

        # shared parameters
        self.node_embeddings = self.ctx_predictor.embeddings
        self.rela_embeddings = self.rela_model.rela_embeddings
        self.num_rela = self.rela_embeddings.num_embeddings
        self.rela_mat = self.rela_embeddings.weight  # [K, D]

        self.tuple_fc_layer = nn.Sequential(nn.Linear(self.embed_dim * 3, self.embed_dim),
                                            nn.Tanh())

        self.tuple_att_layer = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                             nn.Tanh(),
                                             nn.Linear(self.embed_dim, 1))

        # self.tuple_att_layer = nn.Linear(self.embed_dim, 1)

        self.out_layer = nn.Linear(self.embed_dim, 1)

        self.triple_threshold = args.triple_score

    def batch_tuple_represent(self, batch_A, batch_B, target_masks=None):
        '''
        :param batch_A: [B, W, D]
        :param batch_B: [B, V, D]
        :param head_idx: [B, W]
        :param tail_idx: [B, V]
        :param rela_idx: [1]
        :return:
        '''
        batch_size = batch_A.shape[0]
        len_A = batch_A.shape[1]
        len_B = batch_B.shape[1]

        # scoring
        repeat_A = batch_A.repeat(1, 1, len_B).reshape(batch_size, len_A * len_B, -1)  # [B, W*V, D]
        repeat_B = batch_B.repeat(1, len_A, 1)  # [B, W*V, D]

        # repeat to match relation set
        re_repeat_A = repeat_A.repeat(1, 1, self.num_rela).reshape(batch_size, len_A * len_B * self.num_rela, -1)  # [B, W*V*K, D]
        re_repeat_B = repeat_B.repeat(1, 1, self.num_rela).reshape(batch_size, len_A * len_B * self.num_rela, -1)  # [B, W*V*k, D]
        repeat_rela = self.rela_mat.unsqueeze(0).repeat(1, len_A * len_B, 1)  # [1, W*V*K, D]
        # scores = re_repeat_A * repeat_rela * re_repeat_B   # different scoring function
        # scores = scores.sum(dim=-1)  # [B, W*V*K]
        scores = self.rela_model.scoring_func(re_repeat_A, re_repeat_B, repeat_rela)  # [B, W*V*K]

        scores = scores.reshape(batch_size, len_A * len_B, self.num_rela)  # [B, W*V, K]
        if self.args.target_masking:
            scores = scores.masked_fill_(target_masks, -1e10)

        null_scores = scores[:, :, -1].unsqueeze(2).repeat(1, 1, self.num_rela)
        if self.args.cut_rela_score:
            scores = scores.masked_fill_(torch.le(scores, null_scores), -1e10)

        # new_scores = torch.gt(scores, null_scores).float() * scores
        norm_scores = F.softmax(scores, dim=-1)  # [B, W*V, K]

        rela_vec = torch.matmul(norm_scores, self.rela_mat)  # [B, W*V, D]
        tuple_vec = torch.cat((repeat_A, repeat_B, rela_vec), dim=-1)  # [B, W*V, 3*D]
        # tuple_vec = self.tuple_fc_layer(tuple_vec)  # [B, W*V, D]
        tuple_vec = self.tuple_fc_layer(tuple_vec)  # [B, W*V, D]

        return tuple_vec, norm_scores

    def batch_tuple_attention(self, tuple_vec):
        # tuple_vec: [B, W*V, D]
        tuple_att_scores = self.tuple_att_layer(tuple_vec)  # [B, W*V, 1]
        # print(tuple_att_scores)

        if self.args.cut_triple_score:
            tuple_att_scores = tuple_att_scores.masked_fill_(tuple_att_scores <= self.triple_threshold, -1e10)

        # tuple_att_weights = self.tuple_att_layer(att_inputs)  # [B, W*V, 1]
        tuple_att_norms = F.softmax(tuple_att_scores, dim=1)  # [B, W*V, 1]
        tuple_fuse = torch.bmm(tuple_att_norms.permute(0, 2, 1), tuple_vec).squeeze(1)  # [B, D]
        logits = self.out_layer(tuple_fuse)

        return logits, tuple_att_norms

    def forward(self, head_ctx, tail_ctx, target_masks=None):
        # list of pairs: (head_ctx, tail_ctx)
        head_ctx_embed = self.node_embeddings(head_ctx)  # [B, L, D]
        tail_ctx_embed = self.node_embeddings(tail_ctx)  # [B, L, D]

        tuple_vec, norm_scores = self.batch_tuple_represent(head_ctx_embed, tail_ctx_embed, target_masks)  # [B, W*V, D]
        logits, tuple_att_weights = self.batch_tuple_attention(tuple_vec)  # [B, 1]
        logits = logits.squeeze(-1)

        # return logits, (tuple_att_weights.squeeze(), norm_scores)
        return logits, (tuple_att_weights, norm_scores)
