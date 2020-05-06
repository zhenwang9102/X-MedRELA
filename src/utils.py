import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import loader

# class MiniBatcher:
#     def __init__(self):
#         super(MiniBatcher, self).__init__()
#
#     def next_batch(self, inputs):
#         return


def get_random_evidences(pair_data, all_evidences, args):
    evd_dict = {}
    all_evidences = [x for x in all_evidences if x[-1] != 'n/a']
    for idx, pair in enumerate(pair_data):
        t1 = args.node_to_id[pair[0]]
        t2 = args.node_to_id[pair[1]]
        shuffle_idx = np.arange(len(all_evidences))
        np.random.shuffle(shuffle_idx)
        evd_list = []
        for cur_idx in shuffle_idx:
            if len(evd_list) > args.num_triples:
                break
            p1 = args.node_to_id[all_evidences[cur_idx][0]]
            p2 = args.node_to_id[all_evidences[cur_idx][1]]
            r = args.rela_to_id[all_evidences[cur_idx][2]]
            if p1 != t1 and p2 != t2 and p1 != p2:
                evd_list.append((p1, p2, r))

        evd_dict[(t1, t2)] = evd_list[:args.num_triples]

    return evd_dict


def get_evidences(pair_data, ctx_predictor, sel_index, evidence_dict, all_evidences, args):
    evd_dict = {}
    exist_evid_set = set(evidence_dict.keys())
    exist_evid_list = list(evidence_dict.keys())
    all_terms = list(set([x[0] for x in pair_data] + [x[1] for x in pair_data]))
    all_termids = [args.node_to_id[x] for x in all_terms]
    # print('Current # terms: ', len(all_termids))
    terms = torch.tensor(all_termids, device=args.device)
    batch_sel_idx = sel_index.repeat(len(all_termids), 1)
    term_kb_ctx = F.softmax(torch.gather(ctx_predictor(terms), 1, batch_sel_idx), dim=-1)  # [batch, #KB]
    if args.cuda:
        sel_idx_arr = sel_index.to('cpu').detach().data.numpy()
        term_kb_ctx = term_kb_ctx.to('cpu').detach().data.numpy()
    else:
        sel_idx_arr = sel_index.detach().data.numpy()
        term_kb_ctx = term_kb_ctx.detach().data.numpy()

    head_evids_idx = np.array([np.where(sel_idx_arr == args.node_to_id[x[0]]) for x in all_evidences])
    tail_evids_idx = np.array([np.where(sel_idx_arr == args.node_to_id[x[1]]) for x in all_evidences])

    for idx, pair in enumerate(pair_data):
    # for p in tqdm(range(len(pair_data))):
    #     pair = pair_data[p]
        t1 = args.node_to_id[pair[0]]
        t2 = args.node_to_id[pair[1]]
        head_ctx_kb = term_kb_ctx[all_termids.index(t1), :]
        tail_ctx_kb = term_kb_ctx[all_termids.index(t2), :]
        head_evid_score = head_ctx_kb[head_evids_idx].reshape(-1)
        tail_evid_score = tail_ctx_kb[tail_evids_idx].reshape(-1)
        head_tail_sim = head_evid_score * tail_evid_score
        req_evid_idx = np.argsort(head_tail_sim)[::-1]
        # print(req_evid_idx)

        evd_list = []
        for cur_idx in req_evid_idx:
            if len(evd_list) > args.num_triples:
                break
            p1 = args.node_to_id[all_evidences[cur_idx][0]]
            p2 = args.node_to_id[all_evidences[cur_idx][1]]
            if p1 != t1 and p2 != t2 and p1 != p2 and (p1, p2) in exist_evid_set:
                evd_list += [(p1, p2, r) for r in evidence_dict[(p1, p2)]]

        # print(evd_list)
        assert len(evd_list) >= args.num_triples
        evd_dict[(t1, t2)] = evd_list[:args.num_triples]

    return evd_dict


def get_evidences_fix_context(pair_data, ctx_predictor, sel_index, evidences, args):
    evd_dict = {}
    exist_evid = set(evidences.keys())
    all_terms = list(set([x[0] for x in pair_data] + [x[1] for x in pair_data]))
    all_termids = [args.node_to_id[x] for x in all_terms]
    print('Current # terms: ', len(all_termids))
    terms = torch.tensor(all_termids, device=args.device)
    batch_sel_idx = sel_index.repeat(len(all_termids), 1)
    term_kb_ctx = F.softmax(torch.gather(ctx_predictor(terms), 1, batch_sel_idx), dim=-1)  # [batch, #KB]
    if args.cuda:
        sel_idx_arr = sel_index.to('cpu').detach().data.numpy()
        term_kb_ctx = term_kb_ctx.to('cpu').detach().data.numpy()
    else:
        sel_idx_arr = sel_index.detach().data.numpy()
        term_kb_ctx = term_kb_ctx.detach().data.numpy()

    for p in tqdm(range(len(pair_data))):
    # for idx, pair in enumerate(pair_data):
        pair = pair_data[p]
        t1 = args.node_to_id[pair[0]]
        t2 = args.node_to_id[pair[1]]
        head_ctx_kb = term_kb_ctx[all_termids.index(t1), :][:args.num_contexts].reshape(1, -1)
        tail_ctx_kb = term_kb_ctx[all_termids.index(t2), :][:args.num_contexts].reshape(1, -1)
        sim_mat = head_ctx_kb.T * tail_ctx_kb  # [#KB, #KB]
        print(sim_mat.shape)
        exit()
        length = sim_mat.shape[0]
        ctx_sim_idx = np.argsort(sim_mat.reshape(-1))[::-1]
        head_idx = ctx_sim_idx // length
        tail_idx = ctx_sim_idx % length

        sel_idx_arr = sel_idx_arr.reshape(-1)
        evd_list = []
        for i in range(ctx_sim_idx.shape[0]):
            p1 = sel_idx_arr[head_idx[i]]
            p2 = sel_idx_arr[tail_idx[i]]
            if p1 != t1 and p2 != t2 and p1 != p2 and (p1, p2) in exist_evid:
                evd_list += [(p1, p2, r) for r in evidences[(p1, p2)]]

        print(evd_list)
        evd_dict[(t1, t2)] = evd_list[:args.num_triples]

    return evd_dict


def get_evidences_fix_triple(pair_data, ctx_predictor, sel_index, evidences, args):
    evd_dict = {}
    exist_evid = set(evidences.keys())
    all_terms = list(set([x[0] for x in pair_data] + [x[1] for x in pair_data]))
    all_termids = [args.node_to_id[x] for x in all_terms]
    print('Current # terms: ', len(all_termids))
    terms = torch.tensor(all_termids, device=args.device)
    batch_sel_idx = sel_index.repeat(len(all_termids), 1)
    term_kb_ctx = F.softmax(torch.gather(ctx_predictor(terms), 1, batch_sel_idx), dim=-1)  # [batch, #KB]
    if args.cuda:
        sel_idx_arr = sel_index.to('cpu').detach().data.numpy()
        term_kb_ctx = term_kb_ctx.to('cpu').detach().data.numpy()
    else:
        sel_idx_arr = sel_index.detach().data.numpy()
        term_kb_ctx = term_kb_ctx.detach().data.numpy()

    for p in tqdm(range(len(pair_data))):
    # for idx, pair in enumerate(pair_data):
        pair = pair_data[p]
        t1 = args.node_to_id[pair[0]]
        t2 = args.node_to_id[pair[1]]
        # t1 = torch.tensor([t1], device=args.device)
        # t2 = torch.tensor([t2], device=args.device)
        # head_ctx_kb = F.softmax(torch.gather(ctx_predictor(t1), 1, batch_sel_idx), dim=-1)  # [1, #ALL] -> [1, #KB]
        # tail_ctx_kb = F.softmax(torch.gather(ctx_predictor(t2), 1, batch_sel_idx), dim=-1)  # [1, #ALL] -> [1, #KB]
        head_ctx_kb = term_kb_ctx[all_termids.index(t1), :][:1000].reshape(1, -1)
        tail_ctx_kb = term_kb_ctx[all_termids.index(t2), :][:1000].reshape(1, -1)
        # if args.cuda:
        #     sel_idx_arr = batch_sel_idx.to('cpu').detach().data.numpy()
        #     head_ctx_kb = head_ctx_kb.to('cpu').detach().data.numpy()
        #     tail_ctx_kb = tail_ctx_kb.to('cpu').detach().data.numpy()
        # else:
        #     sel_idx_arr = batch_sel_idx.detach().data.numpy()
        #     head_ctx_kb = head_ctx_kb.detach().data.numpy()
        #     tail_ctx_kb = tail_ctx_kb.detach().data.numpy()
        sim_mat = head_ctx_kb.T * tail_ctx_kb  # [#KB, #KB]
        length = sim_mat.shape[0]
        # il1 = np.tril_indices(length)
        # sim_mat[il1] = 0
        ctx_sim_idx = np.argsort(sim_mat.reshape(-1))[::-1]
        head_idx = ctx_sim_idx // length
        tail_idx = ctx_sim_idx % length

        sel_idx_arr = sel_idx_arr.reshape(-1)
        evd_list = []
        for i in range(ctx_sim_idx.shape[0]):
            if len(evd_list) >= args.num_triples:
                break
            p1 = sel_idx_arr[head_idx[i]]
            p2 = sel_idx_arr[tail_idx[i]]
            if p1 != t1 and p2 != t2 and p1 != p2 and (p1, p2) in exist_evid:
                evd_list += [(p1, p2, r) for r in evidences[(p1, p2)]]

        # print(evd_list)
        evd_dict[(t1, t2)] = evd_list[:args.num_triples]

    return evd_dict


def filter_context(heads, tails, ctx_predictor, sel_index, num_contexts):
    '''
    :param heads: [B]
    :param tails: [B]
    :param ctx_predictor: MODEL
    :param sel_index: [All_KB]
    :param num_contexts: Scalar
    :return:
    '''
    batch_size = heads.shape[0]
    batch_sel_idx = sel_index.repeat(batch_size, 1)
    head_ctx_kb = torch.gather(ctx_predictor(heads), 1, batch_sel_idx)  # [B, ALL] -> [B, KB]
    tail_ctx_kb = torch.gather(ctx_predictor(tails), 1, batch_sel_idx)  # [B, ALL] -> [B, KB]
    _, head_kb_idx = torch.topk(head_ctx_kb, num_contexts)
    _, tail_kb_idx = torch.topk(tail_ctx_kb, num_contexts)

    select_head_ctx = torch.gather(batch_sel_idx, 1, head_kb_idx)  # [B, L]
    select_tail_ctx = torch.gather(batch_sel_idx, 1, tail_kb_idx)  # [B, L]

    return select_head_ctx, select_tail_ctx

# def filter_context(heads, tails, ctx_predictor, sel_index, num_contexts, forw_rela_id, back_rela_id, args):
#     '''
#     :param heads: [B]
#     :param tails: [B]
#     :param ctx_predictor: MODEL
#     :param sel_index: [All_KB]
#     :param num_contexts: Scalar
#     :return:
#     '''
#     batch_size = heads.shape[0]
#     batch_sel_idx = sel_index.repeat(batch_size, 1)
#     head_ctx_kb = torch.gather(ctx_predictor(heads), 1, batch_sel_idx)  # [B, ALL] -> [B, KB]
#     tail_ctx_kb = torch.gather(ctx_predictor(tails), 1, batch_sel_idx)  # [B, ALL] -> [B, KB]
#     _, head_kb_idx = torch.topk(head_ctx_kb, num_contexts)
#     _, tail_kb_idx = torch.topk(tail_ctx_kb, num_contexts)
#
#     select_head_ctx = torch.gather(batch_sel_idx, 1, head_kb_idx)  # [B, L]
#     select_tail_ctx = torch.gather(batch_sel_idx, 1, tail_kb_idx)  # [B, L]
#
#     head_idx = torch.nonzero(select_head_ctx == heads.unsqueeze(1))  # [<=B, 2]
#     tail_idx = torch.nonzero(select_tail_ctx == tails.unsqueeze(1))  # [<=B, 2]
#     if head_idx.nelement() == 0 or tail_idx.nelement() == 0:
#         target_forw_idx = torch.tensor([])
#     else:
#         head_target = torch.zeros((batch_size, 1)).to(args.device)
#         tail_target = torch.zeros((batch_size, 1)).to(args.device)
#         print(head_idx.shape)
#         print(head_target.shape)
#         head_target[head_idx[:, 0]] = head_idx[:, 1].reshape(-1, 1).type(torch.float) + 1
#         tail_target[tail_idx[:, 0]] = tail_idx[:, 1].reshape(-1, 1).type(torch.float) + 1
#         filter_index = head_target * tail_target
#         target_idx_idx = torch.nonzero(filter_index)
#         target_index = head_target * num_contexts + tail_target
#         target_final_idx = torch.gather(target_index, 0, target_idx_idx[:, 0].unsqueeze(1).reshape(-1, 1))
#         target_final_idx -= num_contexts + 1
#         target_forw_idx = torch.cat((target_idx_idx[:, 0].reshape(-1, 1),
#                                      target_final_idx.type(torch.long),
#                                      forw_rela_id.expand(1, 1).repeat(target_final_idx.shape[0], 1)), dim=1)
#         print(target_forw_idx.shape)
#
#
#     # reverse relation
#     head_idx = torch.nonzero(select_head_ctx == tails.unsqueeze(1))  # [<=B, 1]
#     tail_idx = torch.nonzero(select_tail_ctx == heads.unsqueeze(1))  # [<=B, 1]
#     if head_idx.nelement() == 0 or tail_idx.nelement() == 0:
#         target_back_idx = torch.tensor([])
#     else:
#         head_target = torch.zeros((batch_size, 1)).to(args.device)
#         tail_target = torch.zeros((batch_size, 1)).to(args.device)
#         head_target[head_idx[0, :]] = head_idx[:, 1].reshape(-1, 1).type(torch.float) + 1
#         tail_target[tail_idx[0, :]] = tail_idx[:, 1].reshape(-1, 1).type(torch.float) + 1
#         filter_index = head_target * tail_target
#         target_idx_idx = torch.nonzero(filter_index)
#         target_index = head_target * num_contexts + tail_target
#         target_final_idx = torch.gather(target_index, 0, target_idx_idx[:, 0].unsqueeze(1).reshape(-1, 1))
#         target_final_idx -= num_contexts + 1
#         target_back_idx = torch.cat((target_idx_idx[:, 0].reshape(-1, 1),
#                                      target_final_idx.type(torch.long),
#                                      back_rela_id.expand(1, 1).repeat(target_final_idx.shape[0], 1)), dim=1)
#
#     target_final_idx = torch.cat((target_forw_idx, target_back_idx), 0)
#
#     return select_head_ctx, select_tail_ctx, target_final_idx


def filter_target_pairs(heads, head_ctx, tails, tail_ctx, forw_rela_id, back_rela_id, args):
    if args.cuda:
        heads = heads.to('cpu').detach().data.numpy()
        head_ctx = head_ctx.to('cpu').detach().data.numpy()
        tails = tails.to('cpu').detach().data.numpy()
        tail_ctx = tail_ctx.to('cpu').detach().data.numpy()

    else:
        heads = heads.detach().data.numpy()
        head_ctx = head_ctx.detach().data.numpy()
        tails = tails.detach().data.numpy()
        tail_ctx = tail_ctx.detach().data.numpy()

    batch_size = heads.shape[0]
    target_final_idx = np.array([])
    # forward relation
    target_head_idx = np.argwhere(head_ctx == heads)  # [<=B, 1]
    target_tail_idx = np.argwhere(tail_ctx == tails)
    if target_head_idx.size == 0 or target_tail_idx.size == 0:
        target_forward_index = np.array([])
    else:
        head_targets = np.zeros((batch_size, 1))
        tail_targets = np.zeros((batch_size, 1))
        head_targets[target_head_idx[:, 0]] = target_head_idx[:, 1].reshape(-1, 1) + 1
        tail_targets[target_tail_idx[:, 0]] = target_tail_idx[:, 1].reshape(-1, 1) + 1
        filter_idx = head_targets * tail_targets
        target_1_index = np.argwhere(filter_idx != 0)[:, 0]  # [<=B]
        # print(filter_idx.shape)
        target_2_index = head_targets * head_ctx.shape[1] + tail_targets
        target_2_index = target_2_index[filter_idx != 0]  # [<=B]
        target_2_index -= num_contexts + 1
        target_3_index = np.array([forw_rela_id] * target_2_index.shape[0])
        target_forward_index = np.concatenate((target_1_index.reshape(-1, 1),
                                               target_2_index.reshape(-1, 1),
                                               target_3_index.reshape(-1, 1)), axis=1)
    target_final_idx = np.vstack([target_final_idx, target_forward_index]) \
        if target_forward_index.size else target_final_idx

    target_head_idx = np.argwhere(head_ctx == tails)  # [<=B, 1]
    target_tail_idx = np.argwhere(tail_ctx == heads)
    if target_head_idx.size == 0 or target_tail_idx.size == 0:
        target_backward_index = np.array([])
    else:
        head_targets = np.zeros((batch_size, 1))
        tail_targets = np.zeros((batch_size, 1))
        head_targets[target_head_idx[:, 0]] = target_head_idx[:, 1].reshape(-1, 1) + 1
        tail_targets[target_tail_idx[:, 0]] = target_tail_idx[:, 1].reshape(-1, 1) + 1
        filter_idx = head_targets * tail_targets
        target_1_index = np.argwhere(filter_idx != 0)[:, 0]  # [<=B]
        # print(filter_idx.shape)
        target_2_index = head_targets * head_ctx.shape[1] + tail_targets
        target_2_index = target_2_index[filter_idx != 0]  # [<=B]
        target_2_index -= num_contexts + 1
        target_3_index = np.array([back_rela_id] * target_2_index.shape[0])
        target_backward_index = np.concatenate((target_1_index.reshape(-1, 1),
                                                target_2_index.reshape(-1, 1),
                                                target_3_index.reshape(-1, 1)), axis=1)

    target_final_idx = np.vstack([target_final_idx, target_backward_index]) \
        if target_backward_index.size else target_final_idx

    mask_mat = np.zeros((batch_size, head_ctx.shape[1] * tail_ctx.shape[1], len(args.rela_to_id)))
    idx = target_final_idx.astype(np.int16)
    if idx.size != 0:
        mask_mat[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    return mask_mat


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pad_sequence(list_ids, min_length=0, max_length=None, padder=0):
    if not max_length:
        max_length = max([len(x) for x in list_ids] + [min_length])
    # print(max_length)
    new_list_ids = []
    for ids in list_ids:
        new_list_ids.append(ids + [padder] * (max_length - len(ids)))
    # return torch.tensor(new_list_ids)
    return np.array(new_list_ids)


def save(model, save_dir, save_prefix, epoch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{0}_epoch_{1}.pt'.format(save_prefix, epoch)
    torch.save(model.state_dict(), save_path)

