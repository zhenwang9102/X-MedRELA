import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from torch.utils.data import DataLoader, Dataset

import utils
import loader


def eval_evidence_binary_relation(dataloader, model, criterion, args):
    with torch.no_grad():
        model.eval()

        test_logits = []
        test_labels = []
        for heads, tails, labels, head_ctx, tail_ctx, rela_ctx in dataloader:

            head_ctx, tail_ctx, rela_ctx = head_ctx.to(args.device), tail_ctx.to(args.device), rela_ctx.to(args.device)
            labels = labels.type(torch.FloatTensor).to(args.device)
            logits, _ = model(head_ctx, tail_ctx, rela_ctx)

            if args.cuda:
                logits = logits.to('cpu').detach().data.numpy()
                labels = labels.to('cpu').detach().data.numpy()
            else:
                logits = logits.detach().data.numpy()
                labels = labels.detach().data.numpy()

            # print(logits.shape)
            test_logits.append(logits)
            test_labels.append(labels)

        test_golds = np.concatenate(test_labels)
        test_logits = np.concatenate(test_logits)
        test_preds = np.where(utils.sigmoid(test_logits) >= 0.5, 1, 0)

        test_acc = metrics.accuracy_score(test_golds, test_preds)
        test_f1 = metrics.f1_score(test_golds, test_preds)
        test_prec = metrics.precision_score(test_golds, test_preds)
        test_reca = metrics.recall_score(test_golds, test_preds)

    return [test_acc, test_f1, test_prec, test_reca]


def eval_binary_relation(dataloader, model, criterion, args):
    with torch.no_grad():
        model.eval()

        test_logits = []
        test_labels = []
        for heads, tails, labels in dataloader:

            heads, tails = heads.to(args.device), tails.to(args.device)

            head_ctx, tail_ctx = utils.filter_context(heads, tails, model.ctx_predictor, args.sel_index, args.num_contexts)

            target_masks = utils.filter_target_pairs(heads, head_ctx, tails, tail_ctx, args.forw_rela, args.back_rela, args)
            target_masks = torch.tensor(target_masks, dtype=torch.bool).to(args.device)

            labels = labels.type(torch.FloatTensor).to(args.device)
            logits, _ = model(head_ctx, tail_ctx, target_masks)

            if args.cuda:
                logits = logits.to('cpu').detach().data.numpy()
                labels = labels.to('cpu').detach().data.numpy()
            else:
                logits = logits.detach().data.numpy()
                labels = labels.detach().data.numpy()

            # print(logits.shape)
            test_logits.append(logits)
            test_labels.append(labels)

        test_golds = np.concatenate(test_labels)
        test_logits = np.concatenate(test_logits)
        test_preds = np.where(utils.sigmoid(test_logits) >= 0.5, 1, 0)

        test_acc = metrics.accuracy_score(test_golds, test_preds)
        test_f1 = metrics.f1_score(test_golds, test_preds)
        test_prec = metrics.precision_score(test_golds, test_preds)
        test_reca = metrics.recall_score(test_golds, test_preds)

    return [test_acc, test_f1, test_prec, test_reca]
