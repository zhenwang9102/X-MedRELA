import os
import sys
import argparse


def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]


def Config():
    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--data_dir', type=str, default='../../data_relation/all_term_term_labels.pkl')
    parser.add_argument('--graph_file', type=str, default='../data/sub_neighbors_dict_ppmi_perBin_1.pkl')
    parser.add_argument('--rela_tp_file', type=str, default='../data/final_relation_tuples_2.pkl')
    parser.add_argument('--bi_rela_file', type=str, default='../final_data/treatment_dataset_neg_1.pkl')
    parser.add_argument('--rela_list_file', type=str, default='../data/umls_relations_2.txt')
    parser.add_argument('--evidence_file', type=str, default='')

    # parser.add_argument('--num_oov', type=int, default=2000)
    # parser.add_argument('--re_sample_test', type='bool', default=False)
    # parser.add_argument('--train_neg_num', type=int, default=50)
    # parser.add_argument('--test_neg_num', type=int, default=100)
    # parser.add_argument('--max_contexts', type=int, default=1000, help='max contexts to look at')
    # parser.add_argument('--context_gamma', type=float, default=0.5)

    # model parameters
    parser.add_argument('--kb_model', type=str, default='transE', help='distmult, transE, ...')
    parser.add_argument('--trans_score', type=str, default='l1', help='l1, l2')
    parser.add_argument('--detach_node_embed', type='bool', default=False, help='detach node embedding in rela_model')
    parser.add_argument('--path_rnn', type=str, default='rnn', help='rnn, lstm')
    parser.add_argument('--add_hloss', type='bool', default=False, help='add entropy loss to attention weight')

    parser.add_argument('--detach_kb_node', type='bool', default=False, help='detach the node embeddings from KB model')
    parser.add_argument('--filter_type', type='bool', default=False, help='filter easily-rejected triples based on types')
    parser.add_argument('--cut_rela_score', type='bool', default=False)
    parser.add_argument('--cut_triple_score', type='bool', default=False)
    parser.add_argument('--triple_score', type=float, default=0.01, help='threshold for triple scores')
    parser.add_argument('--binary_rela', type=str, default='treatment', help='')
    # hyper-parameters
    parser.add_argument('--rela_embed_dim', type=int, default=128)
    parser.add_argument('--node_embed_dim', type=int, default=128)
    parser.add_argument("--num_contexts", type=int, default=50, help="# contexts for interaction")
    parser.add_argument('--num_neg_sample', type=int, default=10)
    parser.add_argument('--num_triples', type=int, default=100, help='# triples to aggregate')

    parser.add_argument('--ctx_model_path', type=str, default='')
    # parser.add_argument('--reload_evidence', type='bool', default=False)
    parser.add_argument('--pretrain_epoch', type=int, default=990)
    parser.add_argument('--random_evidences', type='bool', default=False)
    parser.add_argument('--reload_rand_evidence', type='bool', default=False)

    parser.add_argument('--pretrain_ctx_embed', type='bool', default=False)
    parser.add_argument('--continue_ctx_embed', type='bool', default=False)

    # optim parameters
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of epochs for training")
    parser.add_argument("--log_interval", type=int, default=100, help='step interval for log')
    parser.add_argument("--test_interval", type=int, default=1, help='epoch interval for testing')
    parser.add_argument("--early_stop_epochs", type=int, default=10)
    parser.add_argument("--metric", type=str, default='map', help='mrr or map')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--min_epochs', type=int, default=100, help='minimum number of epochs')
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')

    parser.add_argument('--cooc_batch_size', type=int, default=128)
    parser.add_argument('--rela_batch_size', type=int, default=128)
    parser.add_argument('--cooc_num_workers', type=int, default=0)
    parser.add_argument('--rela_num_workers', type=int, default=0)

    # model selection
    # parser.add_argument('--use_context', type='bool', default=True)
    # parser.add_argument('--do_ctx_interact', type='bool', default=True)
    # parser.add_argument('--use_aux_loss', type='bool', default=False)
    # parser.add_argument('--model', type=str, default='bi', help='bi, diagbi, ntn, dist, ...')

    # path to external files
    parser.add_argument("--embed_filename", type=str, default='../../pretrain_embeddings/glove.6B.100d.txt')
    parser.add_argument('--node_embed_path', type=str, default='../../pretrain_embeddings/line2nd_ttcooc_embedding.txt')
    parser.add_argument('--ngram_embed_path', type=str, default='./data/charNgram.txt')
    # parser.add_argument('--restore_para_file', type=str, default='./final_pretrain_cnn_model_parameters.pkl')
    parser.add_argument('--restore_model_path', type=str, default='')
    parser.add_argument('--restore_idx_data', type=str, default='')
    # parser.add_argument("--logging", type='bool', default=False)
    # parser.add_argument("--log_name", type=str, default='empty.txt')
    parser.add_argument('--restore_model_epoch', type=int, default=600)
    parser.add_argument('--stored_model_path', type=str, default='')

    parser.add_argument("--save_best", type='bool', default=True, help='save model in the best epoch or not')
    parser.add_argument("--save_dir", type=str, default='./saved_models', help='save model in the best epoch or not')
    parser.add_argument("--save_interval", type=int, default=5, help='intervals for saving models')

    parser.add_argument('--num_top_tuples', type=int, default=3)
    parser.add_argument('--num_top_rela', type=int, default=3)
    parser.add_argument('--num_output', type=int, default=10000000)
    # parser.add_argument('--train_single', type='bool')
    # parser.add_argument('--normalize', type='bool', default=False)
    # parser.add_argument('--random_test', type='bool', default=True)

    parser.add_argument('--target_masking', type='bool', default=True)

    args = parser.parse_args()
    print('args: ', args)
    print(os.getpid())

    return args
