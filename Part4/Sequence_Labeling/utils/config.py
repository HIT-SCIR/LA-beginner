# -*- coding: utf-8 -*-#
import argparse
import torch

parser = argparse.ArgumentParser(description='BiLSTM model for sequence labeling')

# Dataset and Other Parameters
parser.add_argument('--data_dir', '-dd', help='dataset file path', type=str, default='./data/ATIS')
parser.add_argument('--save_dir', '-sd', type=str, default='./save/ATIS')
parser.add_argument("--random_state", '-rs', help='random seed', type=int, default=72)
parser.add_argument('--gpu', '-g', action='store_true', help='use gpu', required=False, default=False)

# Training parameters.
parser.add_argument('--num_epoch', '-ne', type=int, default=50)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.4)
parser.add_argument('--early_stop', action='store_true', default=False)
parser.add_argument('--patience', '-pa', type=int, default=10)
parser.add_argument('--alpha', '-alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

# Model parameters.
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=128)

args = parser.parse_args()
args.gpu = args.gpu and torch.cuda.is_available()
print(str(vars(args)))
