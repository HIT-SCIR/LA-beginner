# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import errno
import re
import time
import tempfile
import codecs
import json
import subprocess
import random
import argparse
import logging
from torchnlp.word_to_vector import GloVe


# setting logging
logging.basicConfig(filename='', format='%(asctime)-15s %(levelname)s: %(message)s', level=logging.INFO)

#load structed data splited by '\n',
def load_conll_data(path):
    data_set = []
    label_set = []
    with open(path, 'r') as reader:
        lines = reader.read().split('\n\n')
        for sent in lines:
            words = []
            labels = []
            word_label_pairs = sent.split('\n')
            for word_label_pair in word_label_pairs:
                if word_label_pair.strip():
                    word, label = word_label_pair.strip().split()
                    words.append(word)
                    labels.append(label)
            data_set.append(words)
            label_set.append(labels)
    return data_set, label_set

# load json data (dict or list)
def load_my_json_data(path):
    with open(path, 'r') as reader:
        data_set = []
        label_set = []
        raw_data = json.load(reader)
        for i in raw_data:
            data_set.append(i[0])
            label_set.append(i[1]) # word,label in list
        # data_set = raw_data['seq_ins']
        # label_set = raw_data['seq_outs'] # word,label in dict
    return data_set, label_set

# load train dev and test data from file,by changing '--load_data_type,--train_path,--dev_path,test_path' to choose datatype and path
def load_data(opt):
    train_path = opt.train_path
    dev_path = opt.dev_path
    test_path = opt.test_path
    if opt.load_data_type == 'conll':# conll data
        train_data, train_label = load_conll_data(train_path)
        dev_data, dev_label = load_conll_data(dev_path)
        test_data, test_label = load_conll_data(test_path)
    elif opt.load_data_type == 'json':# json data
        train_data, train_label = load_my_json_data(train_path)
        dev_data, dev_label = load_my_json_data(dev_path)
        test_data, test_label = load_my_json_data(test_path)
    return train_data, train_label, dev_data, dev_label, test_data, test_label


# make word dicts:word2id,label2id.id2label
def make_dict(opt, train_x, train_y=None, dev_y=None, test_y=None):
    word_set = set()
    word2id = {}
    label2id = {}

    # collect the word set
    for sent in train_x:
        for word in sent:
            word_set.add(word)

    # collect label set
    def flatten(l):
        """ list of list to list"""
        return [item for sublist in l for item in sublist]

    def purify(l):
        """ remove B- and I- """
        return set([item.replace('B-','').replace('I-', '') for item in l])

    if opt.label_set_path:
        logging.info('load label from a specific label list file.')
        with open(opt.label_set_path, 'r') as reader:
            label_set = reader.read().strip().split('\n')
    else:
        logging.info('load label from train, dev, test set')
        # get label set without 'B-','I-'
        label_set = list(purify(set(flatten(train_y))) | purify(set(flatten(dev_y))) | purify(set(flatten(test_y))))

    # sort to make embedding id fixed
    word_set = sorted(list(word_set))
    label_set = sorted(label_set)#sort by Dictionary order

    # build 3 dict
    # buid word2id dict
    for word in ['<oov>', '<pad>'] + word_set:
        word2id[word] = len(word2id)

    #build label2id dict
    label2id['<pad>'] = len(label2id)  # place <pad> label in first position if possible
    label2id['O'] = len(label2id)
    for label in label_set:
        #label2id[label] = len(label2id) #without BI
        label2id['B-' + label] = len(label2id)
        label2id['I-' + label] = len(label2id)
    #bulid id2label dict according to label2id dict
    id2label = dict([(idx, label) for label, idx in label2id.items()])

    return word2id, label2id, id2label

#load word embeddings,use Glove here
def load_embedding(opt, word2id):
    if opt.word_embedding:
        logging.info('Load embedding from file.')
        raise NotImplementedError
    else:
        logging.info('Load embedding from pytorch-nlp.')
        if opt.embedding_cache:
            embedding_dict = GloVe(cache=opt.embedding_cache) # load embedding cache from a specific place
        else:
            embedding_dict = GloVe() # load embedding cache from local dir for download now
        logging.info('Load embedding finished.')
        pad_id = word2id['<pad>']
        n_v = len(word2id) # num of words
        n_d = opt.word_dim # emb_dim of word embedding
        embedding_layer = nn.Embedding(n_v, n_d, padding_idx=pad_id) #embedding layer,seehttps://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        embedding_layer.weight.data.uniform_(-0.25, 0.25)
        for word, idx in word2id.items():
            embedding_layer.weight.data[idx] = embedding_dict[word] # use Glove to initialize embedding layer
        logging.info('Word embedding size: {0}'.format(embedding_layer.weight.data.size()))
    return embedding_layer

# create batch
def create_one_batch(x, y, word2id, label2id, sort=True, use_cuda=False, oov='<oov>', pad='<pad>', label_pad='<pad>'):
    batch_size = len(x)
    lst = list(range(batch_size))
    # sort
    if sort:
        lst.sort(key=lambda l: -len(x[l]))
    # get batch word&label
    x = [x[i] for i in lst]
    y = [y[i] for i in lst]
    lens = [len(x[i]) for i in lst]
    max_len = max(lens)
    # get oov\pad\label_pad id
    oov_id, pad_id, label_pad_id = word2id.get(oov, None), word2id.get(pad, None), label2id.get(label_pad, None)
    assert oov_id is not None and pad_id is not None
    # turn batch sentences into tensors and pad sentences in one batch to same len (maxlen)
    batch_x = torch.LongTensor(batch_size, max_len).fill_(pad_id)
    for i, x_i in enumerate(x):
        for j, x_ij in enumerate(x_i):
            batch_x[i][j] = word2id.get(x_ij, oov_id)
    batch_y = torch.LongTensor(batch_size, max_len).fill_(label_pad_id)
    for i, y_i in enumerate(y):
        for j, y_ij in enumerate(y_i):
            batch_y[i][j] = label2id[y_ij]
    # transfer the batch to cuda
    if use_cuda:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
    return batch_x, batch_y, lens

#get all batches
def create_batches(x, y, batch_size, word2id, label2id, sort=True, shuffle=True, use_cuda=False, text=None):
    lst = list(range(len(x)))
    # randomly shuffle batches helps optimeter to work better
    if shuffle:
        random.shuffle(lst)

    if sort:
        lst = sorted(lst, key=lambda i: -len(x[i]))

    x = [x[i] for i in lst]
    y = [y[i] for i in lst]
    text = [text[i] for i in lst] if text is not None else None

    nbatch = (len(x) - 1) // batch_size + 1 # subtract 1 fist to handle situation: len(x) // batch_size == 0
    batches_x, batches_y, batches_lens, batches_text = [], [], [], []
    sum_len = 0.0

    # create all the batches
    for i in range(nbatch):
        start_id, end_id = i * batch_size, (i + 1) * batch_size
        bx, by, blens = create_one_batch(x[start_id: end_id], y[start_id: end_id], word2id, label2id, sort, use_cuda)
        batches_x.append(bx)
        batches_y.append(by)
        batches_lens.append(blens)
        if text is not None:
            batches_text.append(text[start_id: end_id])
        sum_len += sum(blens)

    if shuffle:
        pos_lst = list(range(nbatch))
        random.shuffle(pos_lst)

        batches_x = [batches_x[i] for i in pos_lst]
        batches_y = [batches_y[i] for i in pos_lst]
        batches_text = [batches_text[i] for i in pos_lst]
        batches_lens = [batches_lens[i] for i in pos_lst]
        batches_text = [batches_text[i] for i in pos_lst] if text is not None else None

    logging.info("{} batches, avg len: {:.1f}".format(nbatch, sum_len / len(x)))
    if text is not None:
        return batches_x, batches_y, batches_lens, batches_text
    return batches_x, batches_y, batches_lens

# decoder : linear
class ClassifyLayer(nn.Module):
    def __init__(self, input_size, num_tags, label2id, label_pad='<pad>', use_cuda=False):
        super(ClassifyLayer, self).__init__()
        self.use_cuda = use_cuda # whether to use cuda
        self.num_tags = num_tags # num of labels
        self.label2id = label2id
        self.label_pad = label_pad
        # define decoder neural network
        self.hidden2tag = nn.Linear(in_features=input_size, out_features=num_tags) #linear layer learn details on https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.logsoftmax = nn.LogSoftmax(dim=2) # logsoftmax learn details on https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
        tag_weights = torch.ones(num_tags)
        tag_weights[label2id[label_pad]] = 0
        self.criterion = nn.NLLLoss(tag_weights) # negtive loglikelihood loss learn details on https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        #consider how to use crossentropyloss replace NLLLoss and LogSoftmax ,learn details on https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    def forward(self, x, y):
        """
        :param x: torch.Tensor (batch_size, seq_len, n_in)
        :param y: torch.Tensor (batch_size, seq_len)
        :return:
        """
        tag_scores = self.hidden2tag(x)
        if self.training:
            tag_scores = self.logsoftmax(tag_scores) # we need to calculate probablities of each tags when train for calculate loss in the next step
        if self.label2id[self.label_pad] == 0:
            _, tag_result = torch.max(tag_scores[:, :, 1:], 2)  # block <pad> label as predict output
        else:
            _, tag_result = torch.max(tag_scores, 2)  # give up to block <pad> label for efficiency
        tag_result.add_(1)
        if self.training:
            return tag_result, self.criterion(tag_scores.view(-1, self.num_tags), Variable(y).view(-1)) # computing loss
        else:
            return tag_result, torch.FloatTensor([0.0]) # return predict result

    def get_probs(self, x):
        tag_scores = self.hidden2tag(x)
        if self.training:
            tag_scores = self.logsoftmax(tag_scores)

        return tag_scores

#the whole model  encoder:bilstm,decoder:linear
class Model(nn.Module):
    def __init__(self, opt, embedding_layer, nclass, label2id, use_cuda):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        self.opt = opt
        self.embedding_layer = embedding_layer

        encoder_output_size = None
        # define bilstm encoder
        if opt.encoder.lower() == 'lstm':
            self.encoder = nn.LSTM(
                input_size=opt.word_dim, hidden_size=opt.hidden_dim, num_layers=opt.depth,
                batch_first=True, dropout=opt.dropout, bidirectional=True
            ) # encoder :bilstm encoder (word_dim , hidden_dim*2)
            encoder_output_size = 2 * opt.hidden_dim  # because of the bi-directional
        else:
            raise ValueError('Unknown classifier {0}'.format(opt.lstm))
        # define linear decoder
        if opt.classifier.lower() == 'vanilla':
            self.classify_layer = ClassifyLayer(encoder_output_size, nclass, label2id, use_cuda=use_cuda)
        else:
            raise ValueError('Unknown classifier {0}'.format(opt.classifier))

        self.train_time = 0
        self.eval_time = 0
        self.emb_time = 0
        self.classify_time = 0

    def forward(self, batch_x, batch_y):
        start_time = time.time() # get local time for log
        batch_size, seq_len = batch_x.size(0), batch_x.size(1)
        # get embeddings and move it to cuda
        word_emb = self.embedding_layer(Variable(batch_x).cuda() if self.use_cuda else Variable(batch_x))
        # dropout
        word_emb = F.dropout(word_emb, self.opt.dropout, self.training)

        if not self.training:
            self.emb_time += time.time() - start_time

        start_time = time.time()

        if self.opt.encoder.lower() == 'lstm':
            # put the word embedding through the bilstm encoder
            output, hidden = self.encoder(word_emb)
        else:
            raise ValueError('unknown encoder: {0}'.format(self.opt.encoder))

        if self.training:
            self.train_time += time.time() - start_time
        else:
            self.eval_time += time.time() - start_time
        start_time = time.time()
        # put hidden state from bilstm through linear decoder to compute output and loss
        output, loss = self.classify_layer.forward(output, batch_y)
        if not self.training:
            self.classify_time += time.time() - start_time

        return output, loss

# evaluation trained models
def eval_model(model, valid_x, valid_y, valid_lens, valid_text, id2label, opt):
    if opt.output is not None:
        output_path = opt.output
        fpo = codecs.open(output_path, 'wb', encoding='utf-8')
    else:
        descriptor, output_path = tempfile.mkstemp(suffix='.tmp')
        fpo = codecs.getwriter('utf-8')(os.fdopen(descriptor, 'wb'))

    model.eval() # Sets the module in evaluation mode,turn dropout and batchnorm etc. learn more on https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=eval#torch.nn.Module.eval
    # feed validation data batches to the model to predict
    for x, y, lens, text in zip(valid_x, valid_y, valid_lens, valid_text):
        output, loss = model.forward(x, y)
        output_data = output.data
        # evaluate on each data point
        for bid in range(len(x)):
            for k, (word, true_tag, predict_tag) in enumerate(zip(text[bid], y[bid], output_data[bid])):

                true_tag = id2label[int(true_tag)] # get real tag name

                predict_tag = id2label[int(predict_tag)] # get predict tag name
                print('{1} {2} {3}'.format(k + 1, word, true_tag, predict_tag), file=fpo) #write predict and real label into a file

            print(file=fpo)
    fpo.close()
    # use script to eval
    script_args = ['perl', opt.script]
    # script_args = ['perl', opt.script, '< ', output_path]
    with open(output_path, 'r') as res_file:
        # use commandline tools to run eval script
        p = subprocess.Popen(script_args, stdout=subprocess.PIPE, stdin=res_file)
        # logging.info('Eval script args:{0}'.format(p.args))
        p.wait()

        std_results = p.stdout.readlines() # get results from standard commandline output
        # print('==============\ndebug:{0}\n============='.format(std_results))
        std_results = str(std_results[1]).split()
    # print('========script output======\n{0}================{1}'.format(std_results,
    #       ['perl', opt.script, '–d', '\\t', '<', output_path]))
    precision = float(std_results[3].replace('%;', ''))
    recall = float(std_results[5].replace('%;', ''))
    f1 = float(std_results[7].replace('%;', '').replace("\\n'", ''))
    os.remove(output_path)
    return precision, recall, f1

# train model
def train_model(epoch, model, optimizer,
                train_x, train_y, train_lens,
                valid_x, valid_y, valid_lens, valid_text,
                test_x, test_y, test_lens, test_text,
                ix2label, best_valid, test_f1_score):
    model.train()
    # set model to train mode able dropout etc.learn more on https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=train#torch.nn.Module.train
    opt = model.opt

    total_loss = 0.0
    cnt = 0
    start_time = time.time()

    lst = list(range(len(train_x)))
    random.shuffle(lst)
    train_x = [train_x[l] for l in lst]
    train_y = [train_y[l] for l in lst]
    train_lens = [train_lens[l] for l in lst]
    # training
    for x, y, lens in zip(train_x, train_y, train_lens):
        cnt += 1
        model.zero_grad()
        _, loss = model.forward(x, y) # forward
        total_loss += loss.data.item()
        n_tags = sum(lens)
        loss.backward() # backward
        torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip_grad) # use clip_grad trick, learn more on https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html?highlight=clip_grad_norm#torch.nn.utils.clip_grad_norm_
        optimizer.step() # optimize
        if cnt * opt.batch_size % 1024 == 0:
            logging.info("Epoch={} iter={} lr={:.6f} train_ave_loss={:.6f} time={:.2f}s".format(
                epoch, cnt, optimizer.param_groups[0]['lr'],
                1.0 * loss.item() / n_tags, time.time() - start_time
            ))
            start_time = time.time()
    # eval
    dev_precision, dev_recall, dev_f1_score = eval_model(model, valid_x, valid_y, valid_lens, valid_text,
                              ix2label, opt)
    logging.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}".format(
        epoch, cnt, optimizer.param_groups[0]['lr'], total_loss, dev_f1_score))

    if dev_f1_score > best_valid:
        torch.save(model.state_dict(), os.path.join(opt.model, 'model.pl'))
        best_valid = dev_f1_score # choose the best model on the dev set
        test_precision, test_recall, test_f1_score = eval_model(model, test_x, test_y, test_lens, test_text,
                                   ix2label, opt)
        logging.info("New record achieved!")
        logging.info("Epoch={} iter={} lr={:.6f} test_precision={:.6f}, test_recall={:.6f}, test_f1={:.6f}".format(
            epoch, cnt, optimizer.param_groups[0]['lr'], test_precision, test_recall, test_f1_score))
    return best_valid, test_f1_score

# the whole training and testing procedure
def train_and_test(opt):
    use_cuda = opt.gpu >= 0 and torch.cuda.is_available()
    # load data
    logging.info('Start to load data')
    train_x, train_y, dev_x, dev_y, test_x, test_y = \
        load_data(opt)
    logging.info('training instance: {}, validation instance: {}, test instance: {}.'.format(
        len(train_y), len(dev_y), len(test_y)))

    # create dict for label and word
    word2id, label2id, id2label = make_dict(opt, train_x, train_y, dev_y, test_y)
    nclasses = len(label2id)

    # load & create embedding layer
    embedding_layer = load_embedding(opt, word2id)

    # create batch data
    train_x, train_y, train_lens, train_text = create_batches(
        train_x, train_y, opt.batch_size, word2id, label2id, use_cuda=use_cuda, text=train_x,shuffle=True
    )
    dev_x, dev_y, dev_lens, dev_text = create_batches(
        dev_x, dev_y, opt.batch_size, word2id, label2id, shuffle=True, sort=False, use_cuda=use_cuda, text=dev_x
    )
    test_x, test_y, test_lens, test_text = create_batches(
        test_x, test_y, opt.batch_size, word2id, label2id, shuffle=True, sort=False, use_cuda=use_cuda, text=test_x
    )

    # build model
    model = Model(opt, embedding_layer, nclasses, label2id, use_cuda)
    logging.info(str(model))
    if use_cuda:
        model = model.cuda() # shift the whole model to cuda

    # record configuration
    try:
        os.makedirs(opt.model)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    with open(os.path.join(opt.model, 'word2id.dict'), 'w') as w2id_file:
        json.dump(word2id, w2id_file)
    with open(os.path.join(opt.model, 'label2id.dict'), 'w') as l2id_file:
        json.dump(label2id, l2id_file)
    with open(os.path.join(opt.model, 'config.dict'), 'w') as config_file:
        json.dump(vars(opt), config_file)

    # train and select model
    if opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr) # Adam optimazer, learn more on https://pytorch.org/docs/stable/generated/torch.optim.Adam.html.
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt.lr) #SGD optimazer, learn more on https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        #learn more about optimization functions on https: // pytorch.org / docs / stable / optim.html
    best_valid, test_result = -1e8, -1e8
    for epoch in range(opt.max_epoch):
        best_valid, test_result = train_model(
            epoch, model, optimizer,
            train_x, train_y, train_lens,
            dev_x, dev_y, dev_lens, dev_text,
            test_x, test_y, test_lens, test_text,
            id2label, best_valid, test_result
        ) # training model
        # trick : learning rate decay
        if opt.lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= opt.lr_decay  # there is only one group of parameters, so use index 0

        logging.info('Total encoder time: {:.2f}s'.format(model.eval_time / (epoch + 1)))
        logging.info('Total embedding time: {:.2f}s'.format(model.emb_time / (epoch + 1)))
        logging.info('Total classify time: {:.2f}s'.format(model.classify_time / (epoch + 1)))
    logging.info("best_valid_acc: {:.6f}".format(best_valid))
    logging.info("test_acc: {:.6f}".format(test_result))


def main():
    # set some arguments
    cmd = argparse.ArgumentParser()

    # running mode
    cmd.add_argument('-tt', '--train_and_test', action='store_true', help='run train and test at the same time')

    # define path
    cmd.add_argument('--load_data_type',required=True,help='the data type')
    cmd.add_argument('--train_path', required=True, help='the path to the training file.')
    cmd.add_argument('--dev_path', required=True, help='the path to the validation file.')
    cmd.add_argument('--test_path', required=True, help='the path to the testing file.')
    cmd.add_argument('--label_set_path', default='', help='the path to the file record all label name')
    cmd.add_argument("--model", required=True, help="path to save model,eg: ./model.pkl")
    cmd.add_argument('--output', help='The path to the output file.')
    cmd.add_argument("--script", default='./eval/conlleval.pl', help="The path to the evaluation script")
    cmd.add_argument("--word_embedding", type=str, default='',
                     help="pass a path to word vectors from file(not finished), empty string to load from pytorch-nlp")
    cmd.add_argument("--embedding_cache", type=str, default='',
                     help="path to embedding cache dir. if use pytorch nlp, use this path to avoid downloading")

    # environment setting
    cmd.add_argument('--seed', default=1, type=int, help='the random seed.')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')

    # define detail
    cmd.add_argument('--encoder', default='lstm', choices=['lstm'],
                     help='the type of encoder: valid options=[lstm]')
    cmd.add_argument('--classifier', default='vanilla', choices=['vanilla'],
                     help='The type of classifier: valid options=[vanilla]')
    cmd.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'],
                     help='the type of optimizer: valid options=[sgd, adam]')

    cmd.add_argument("--batch_size", "--batch", type=int, default=128, help='the batch size.')
    cmd.add_argument("--hidden_dim", "--hidden", type=int, default=128, help='the hidden dimension.')
    cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')
    cmd.add_argument("--word_dim", type=int, default=300, help='the input dimension.')
    cmd.add_argument("--dropout", type=float, default=0.5, help='the dropout rate')
    cmd.add_argument("--depth", type=int, default=2, help='the depth of lstm')
    cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
    cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')
    cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad.')

    opt = cmd.parse_args()# get opt from these arguments

    print(opt)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    if opt.gpu >= 0:
        torch.cuda.set_device(opt.gpu)
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    if opt.train_and_test:
        print('Start training.')
        train_and_test(opt)


if __name__ == '__main__':
    main()
