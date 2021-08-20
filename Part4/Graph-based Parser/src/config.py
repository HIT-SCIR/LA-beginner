import argparse

parser = argparse.ArgumentParser()

# general argument
parser.add_argument('--random_seed',
                    '-rs',
                    type=int,
                    default=0,
                    help="random seed")
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='train',
                    help="train model or predict result")
parser.add_argument('--model_path',
                    '-mp',
                    type=str,
                    default='./model',
                    help="path to save model")
parser.add_argument('--max_line',
                    '-ml',
                    type=int,
                    default=5,
                    help="max length to be processed")
parser.add_argument('--batch_size',
                    '-bs',
                    type=int,
                    default=2,
                    help="batch size")
parser.add_argument('--embedding_dim',
                    '-ed',
                    type=int,
                    default=16,
                    help='dimension of embedding layer')
parser.add_argument('--hidden_size', '-hs', type=int, default=32,
                    help='hidden size of BiLSTM and MLP')
parser.add_argument('--layer_num',
                    '-ln',
                    type=int,
                    default=1,
                    help='layer number')
parser.add_argument('--show_tqdm', '-st', action='store_true',
                    default=False, help='whether show tqdm')
parser.add_argument('--core_num',
                    '-cn',
                    type=int,
                    default=1,
                    help="cpu core to be used")
parser.add_argument('--decoder_type',
                    '-dt',
                    type=str,
                    default='eisner',
                    choices=['eisner', 'chuliu'],
                    help='decoder type')


# train mode argument
parser.add_argument('--train_data_path',
                    '-tdp',
                    type=str,
                    default='./data/train.conll',
                    help="path of train data")
parser.add_argument('--valid_data_path',
                    '-vdq',
                    type=str,
                    default='./data/valid.conll',
                    help="path of valid data")
parser.add_argument('--epoch', '-e', type=int, default=200, help="epoch")
parser.add_argument('--learning_rate',
                    '-lr',
                    type=float,
                    default=1e-5,
                    help="learning rate")
parser.add_argument('--dropout_rate',
                    '-dr',
                    type=float,
                    default=0.3,
                    help="dropout")
parser.add_argument('--random_rate',
                    '-rr',
                    type=float,
                    default=0.1,
                    help='rate of edges to remove of second MST')

# test mode argument
parser.add_argument('--predict_data_path',
                    '-pdp',
                    type=str,
                    default='./data/test.conll',
                    help="path of test data")
parser.add_argument('--save_path',
                    '-sp',
                    type=str,
                    default='./data/result.txt',
                    help="output path")

args = parser.parse_args()
