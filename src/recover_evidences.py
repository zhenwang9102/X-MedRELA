import os
import sys
import numpy as np

import torch


def recovering_ranked_triples(head_ctx, tail_ctx, tuple_att, pair_scores, num_top_triples, mapper, sem_net=None):
    # tuple_att: [L]
    # pair_scores: [L, K]
    joint_scores = pair_scores * tuple_att.reshape(-1, 1)  # [L, K]
    pair2idx = {}
    pair_idx = 0
    for idx_h, c_h in enumerate(head_ctx):
        for idx_t, c_t in enumerate(tail_ctx):
            pair2idx[(c_h, c_t)] = pair_idx
            pair_idx += 1
    idx2pair = {v: k for k, v in pair2idx.items()}

    triple2score = {}
    for i in range(joint_scores.shape[0]):
        for j in range(joint_scores.shape[1]):
            cur_pair = idx2pair[i]
            cur_rela = j
            triple2score[(cur_pair, cur_rela)] = joint_scores[i, j]

    top_triples = sorted(triple2score.items(), key=lambda kv: kv[1], reverse=True)[:num_top_triples]
    # print(top_triples)

    triples_str = [(mapper['term2str'][mapper['id2node'][x[0][0][0]]],
                mapper['term2str'][mapper['id2node'][x[0][0][1]]],
                mapper['id2rela'][x[0][1]]) for x in top_triples]

    triples_id = [(mapper['id2node'][x[0][0][0]], mapper['id2node'][x[0][0][1]], x[0][1]) for x in top_triples]

    # print(triples)
    return triples_str, triples_id


def recovering_context_tuples(head_ctx, tail_ctx, tuple_att, pair_scores,
                              num_top_tuples, num_top_rela, mapper, sem_net=None):
    '''
    :param head_ctx: a list of contexts
    :param tail_ctx: a list of contexts
    :param tuple_att: []
    :param pair_scores:
    :param num_top_tuples: 10
    :param num_top_rela: 5
    :param mapper: a list of mapping dict: ['id2node', 'term2str', 'concept2str', 'id2rela', 'term2concept']
    :return:
    '''
    att_dict = {}
    score_dict = {}
    for idx_h, c_h in enumerate(head_ctx):
        for idx_t, c_t in enumerate(tail_ctx):
            att_dict[(c_h, c_t)] = tuple_att[idx_h * len(head_ctx) + idx_t]
            score_dict[(c_h, c_t)] = pair_scores[idx_h * len(head_ctx) + idx_t]

    top_att_pairs = sorted(att_dict.items(), key=lambda kv: kv[1], reverse=True)[:num_top_tuples]
    top_att_pairs = [x[0] for x in top_att_pairs]
    for pair in top_att_pairs:
        c1_termid = mapper['id2node'][pair[0]]
        c2_termid = mapper['id2node'][pair[1]]
        print('{0} --- {1} ({2:.3f})'.format(mapper['term2str'][c1_termid],
                                             mapper['term2str'][c2_termid],
                                             att_dict[pair]))

        scores = score_dict[pair]
        top_rela = np.argsort(scores)[::-1][:num_top_rela]
        top_rela_p = np.sort(scores)[::-1][:num_top_rela]

        if sem_net and c1_termid in mapper['term2concept'] and c2_termid in mapper['term2concept']:
            head_cuis = [mapper['concept2cui'][x] for x in mapper['term2concept'][c1_termid]
                         if mapper['concept2cui'][x] in mapper['cui2type']]
            tail_cuis = [mapper['concept2cui'][x] for x in mapper['term2concept'][c2_termid]
                         if mapper['concept2cui'][x] in mapper['cui2type']]

            if head_cuis == [] or tail_cuis == []:
                # print('\t\tNo type info found!')
                continue
            else:
                head_types = list(set([x for hc in head_cuis for x in mapper['cui2type'][hc]]))
                tail_types = list(set([x for tc in tail_cuis for x in mapper['cui2type'][tc]]))

                for i in range(num_top_rela):
                    cur_relastr = mapper['id2rela'][top_rela[i]]

                    true_rela = 0
                    for hct in head_types:
                        for tct in tail_types:
                            if (hct, tct, cur_relastr) in sem_net:
                                true_rela += 1

                            # if sem_net.has_edge(hct, tct):
                            #     relas = [sem_net[hct][tct][x]['srela'] for x in list(sem_net[hct][tct])]
                            #     # print(relas)
                            #     if cur_relastr in relas:
                            #         true_rela += 1
                    if true_rela:
                        print('\t- {0} ({1:.3f}) {2}'.format(cur_relastr, round(top_rela_p[i], 3), 'T'))
                    else:
                        print('\t- {0} ({1:.3f}) {2}'.format(cur_relastr, round(top_rela_p[i], 3), 'F'))

        else:
            for i in range(num_top_rela):
                print('\t- {0} ({1:.3f})'.format(mapper['id2rela'][top_rela[i]], round(top_rela_p[i], 3)))

    return
