# coding: utf-8

import os
import sys
import json
import torch
import fitlog
import logging
import argparse
import datetime


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt='%Y/%d/%m %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


parser = argparse.ArgumentParser()
# Dataset and Other Parameters
parser.add_argument('--data_dir', '-dd', type=str, default='data/ProSLU')
parser.add_argument('--save_dir', '-sd', type=str, default='save/')
parser.add_argument('--load_dir', '-ld', type=str, default=None)
parser.add_argument('--slot_without_intent', '-swi', action='store_true', required=False)

parser.add_argument('--gpu', '-g', action='store_true', help='Use cuda backend', default=False, required=False)
parser.add_argument("--use_info", '-ui', help='use info', action='store_true', required=False)
parser.add_argument('--model_type', '-mt', type=str, default="ELECTRA")
parser.add_argument('--max_length', '-ml', type=int, help='max length for KG', default=64)
# logging
parser.add_argument('--use_fitlog', '-uf', help='use fitlog', action='store_true', required=False)
parser.add_argument('--fit_log_dir', '-fld', required=False, default='logs/')
parser.add_argument('--logging_steps', '-ls', type=int, default=10)

# Training parameters
# random seed
parser.add_argument("--fix_seed", '-fs', help='fix seed', action='store_true', required=False)
parser.add_argument("--random_seed", '-rs', type=int, default=0)
# early stop
parser.add_argument('--early_stop', '-es', action='store_true', required=False)
parser.add_argument('--patience', '-pa', type=int, default=10)
#
parser.add_argument('--do_eval', '-de', action='store_true', required=False)
parser.add_argument('--num_epoch', '-ne', type=int, default=100)
parser.add_argument('--batch_size', '-bs', type=int, default=8)
# TODO: weight decay
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2, help='weight decay for AdamW')
parser.add_argument('--clip', '-clip', help='gradient clipping', required=False, default=10)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
# parser.add_argument("--node_learning_rate", '-nlr', type=float, default=None)
parser.add_argument("--plm_learning_rate", '-plr', type=float, default=0.00001)
parser.add_argument("--node_plm_learning_rate", '-nplr', type=float, default=None)

parser.add_argument('--lr_scheduler_factor', '-lsf', type=float, default=0.5)
parser.add_argument('--lr_scheduler_patience', '-lsp', type=int, default=5)

parser.add_argument('--dropout_rate', '-dr', type=float, default=0.4)
parser.add_argument('--plm_dropout_rate', '-pdr', type=float, default=0.1)
parser.add_argument("--differentiable", "-d", action="store_true", default=False)
parser.add_argument('--slot_forcing_rate', '-sfr', type=float, default=0.9)

# Model parameters
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=768)
parser.add_argument('--attention_hidden_dim', '-ahd', type=int, default=1024)
parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)
parser.add_argument('--use_pretrained', '-up', action='store_true', help='use pretrained models', required=False)
parser.add_argument('--word_level_pretrained', '-wlp', action='store_true',
                    help='pretrained models encode user utterances in word level', required=False)
parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=64)

# define related parameters for edges and nodes in heterogeneous/homogeneous graph
parser.add_argument('--homogeneous_graph', '-homog', action='store_true', required=False)
# support embed_bow for kg_node_triplet_representation
parser.add_argument('--kg_node_triplet_representation', '-kgntr', type=str, default="encode_bow",
                    choices=["natural", "concat", "encode_bow", "embed_bow"], required=False)
parser.add_argument('--special_node_triplet_representation', '-spntr', type=str, default="embed_bow",
                    choices=["natural", "concat", "embed_bow"], required=False)
parser.add_argument('--kg_global_node', '-kggn', action='store_true', help='use global node in KG', required=False)
parser.add_argument('--up_global_node', '-upgn', action='store_true', help='use global node in UP', required=False)
parser.add_argument('--ca_global_node', '-cagn', action='store_true', help='use global node in CA', required=False)
parser.add_argument('--up_preference_global_node', '-uppgn', action='store_true',
                    help='use preference global node in UP', required=False)
parser.add_argument('--kg_inter_field_edge', '-kgirfe', action='store_true',
                    help="edges among all fields in a subject", required=False)
parser.add_argument('--kg_intra_group_edge', '-kgiage', action='store_true',
                    help="edges among all subjects in a group", required=False)
parser.add_argument('--kg_inter_group_edge', '-kgirge', action='store_true',
                    help="edges among all subjects in different groups", required=False)
parser.add_argument('--up_intra_preference_edge', '-upiape', action='store_true',
                    help="egdes among all fields in a preference", required=False)
parser.add_argument('--up_inter_preference_edge', '-upirpe', action='store_true',
                    help="edges among all preferences in a user profile", required=False)
parser.add_argument('--ca_inter_status_edge', '-cairse', action='store_true',
                    help="edges among all statuses in a context awareness", required=False)
parser.add_argument('--kg_up_edge', '-kgup', action='store_true', help="edges between KG and UP", required=False)
parser.add_argument('--kg_ca_edge', '-kgca', action='store_true', help="edges between KG and CA", required=False)
parser.add_argument('--up_ca_edge', '-upca', action='store_true', help="edges between UP and CA", required=False)
parser.add_argument('--utter_char_edge', '-uttchar', action='store_true',
                    help="edges between user utterance and all characters", required=False)
parser.add_argument('--utter_profile_edge', '-uttpf', type=str, default="none",
                    choices=["none", "global", "leaf", "both"],
                    help="edges between user utterance and global nodes", required=False)

# Model parameters related to graph neural network
parser.add_argument('--graph_dropout_rate', '-gdr', type=float, default=0.2)
parser.add_argument('--node_word_embedding_dim', '-nwed', type=int, default=128)
parser.add_argument('--node_encoder_hidden_dim', '-nehd', type=int, default=768)
parser.add_argument('--node_attention_hidden_dim', '-nahd', type=int, default=1024)
parser.add_argument('--node_attention_output_dim', '-naod', type=int, default=128)
parser.add_argument('--graph_hidden_size', '-ghs', type=int, default=128)
parser.add_argument('--num_graph_layer', '-ngl', type=int, default=2)
parser.add_argument('--num_graph_heads', '-ngh', type=int, default=8)
parser.add_argument('--profile_info_dim', '-pid', type=int, default=128)
parser.add_argument('--disable_node_pretrained', '-dnp', action='store_true',
                    help='disable pretrained models for node representation', required=False)
parser.add_argument('--homogeneous_aggregation_strategy', '-has', type=str, default="batch_weight",
                    choices=["batch_weight", "individual_weight", "sample_weight", "sum", "mean"],
                    help="Aggregation of node representations from multiple homogeneous graphs with the same node but different edge types to obtain the node representation of heterogeneous graphs")
parser.add_argument('--info_utilization_way', '-iuw', type=str, default="node_aggregation",
                    choices=["node_weighting", "node_aggregation"],
                    help="How to utilize the profile information for slot filling and intent detection")
parser.add_argument('--enable_character_node', '-ecn', action='store_true')
parser.add_argument('--num_supporting_node', '-nse', type=int, default=-1,
                    help="-1: all nodes; 0: only leaf nodes; >0: number of supporting nodes")
parser.add_argument('--sentence_level_node_weighting', '-slnw', action='store_true', required=False)
parser.add_argument('--append_null_into_leaf', '-anil', action='store_true', required=False)

args = parser.parse_args()
args.up_keys = ['音视频应用偏好', '出行交通工具偏好', '长途交通工具偏好', '是否有车']
args.ca_keys = ['移动状态', '姿态识别', '地理围栏', '户外围栏']

# args.node_learning_rate = args.learning_rate if args.node_learning_rate is None else args.node_learning_rate
args.node_plm_learning_rate = args.plm_learning_rate if args.node_plm_learning_rate is None else args.node_plm_learning_rate

VERBOSE = True

if not args.use_fitlog:
    fitlog.debug()

timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')

if args.use_pretrained:
    prefix = 'PLM_SLU'
else:
    prefix = 'SLU'
    args.model_type = 'LSTM'
if args.use_info:
    # heterogeneous graph or homogeneous graph
    gn = "HOG" if args.homogeneous_graph else "HEG"
    if args.use_pretrained and not args.disable_node_pretrained:
        gn = f"PLM_{gn}"
    prefix += f'+{gn}'

args.save_dir = os.path.join(
    'save', prefix,
    f"{args.use_info}_{args.model_type}_wlp={args.word_level_pretrained}_{args.batch_size}_"
    f"lrs={args.lr_scheduler_factor}-{args.lr_scheduler_patience}_"
    f"U={args.dropout_rate}-{args.learning_rate}-{args.plm_learning_rate}-{args.word_embedding_dim}_"
    f"G={args.graph_dropout_rate}-{args.node_word_embedding_dim}-{args.profile_info_dim}_"
    f"{timestamp}")

os.makedirs(args.save_dir, exist_ok=True)
log_path = os.path.join(args.save_dir, "config.json")

with open(log_path, "w", encoding="utf8") as fw:
    fw.write(json.dumps(args.__dict__, indent=True))

mylogger = get_logger(os.path.join(args.save_dir, 'log.txt'), name='SLU')
mylogger.info(str(vars(args)))

USE_CUDA = args.gpu
print("USE CUDA Backend: " + str(USE_CUDA))

# logger for metrics and losses
fitlog.set_log_dir(args.fit_log_dir)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)

# maximum number of elements for each graph node
MEM_TOKEN_SIZE = 3

remove_node_after_null = False
homogeneous_aggregation_strategy = args.homogeneous_aggregation_strategy

# Model Dict
if args.model_type != 'LSTM':
    model_type = {
        'RoBERTa': "../ProSLU/PretrainModel/bert/chinese-roberta-wwm-ext",
        'BERT': "../ProSLU/PretrainModel/bert/chinese-bert-wwm-ext",
        'XLNet': "../ProSLU/PretrainModel/bert/chinese-xlnet-base",
        'ELECTRA': "../ProSLU/PretrainModel/bert/chinese-electra-180g-base-discriminator",
    }

    args.model_type_path = model_type[args.model_type]

# model file name for saving
ModelFileName = 'model.pt'
