# -*- coding: utf-8 -*-#
import torch
import torch.nn as nn
from utils.config import args
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import transformers


def ensure_gpu(x):
    if args.gpu:
        return x.cuda()
    return x

class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return padded_hiddens


class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize an bert Encoder object.
        self.__encoder = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.__tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.pad = self.__tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        self.sep = self.__tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        self.cls = self.__tokenizer.convert_tokens_to_ids(["[CLS]"])[0]

    def get_info(self, x):
        """
        :param x: a batch of sentences which has been replaced in some positions.
        :return: encoded elements token_loc, token_ids, type_ids, mask_ids
        """
        token_ids = []
        token_loc = []
        for xx in x:
            per_token_ids = [self.cls]
            per_token_loc = []
            cur_idx = 1
            for token in xx:
                tmp_ids = self.__tokenizer.encode(token)[1:-1]
                per_token_ids += tmp_ids
                per_token_loc.append(cur_idx)
                cur_idx += len(tmp_ids)
            per_token_ids += [self.sep]
            token_ids.append(per_token_ids)
            token_loc.append(per_token_loc)
        lens = [len(p) for p in token_ids]
        max_len = max(lens)
        mask_ids = []
        type_ids = []
        for per_token_ids in token_ids:
            per_mask_ids = [1] * len(per_token_ids) + [0] * (max_len - len(per_token_ids))
            per_token_ids += [self.pad] * (max_len - len(per_token_ids))
            per_type_ids = [0] * max_len
            mask_ids.append(per_mask_ids)
            type_ids.append(per_type_ids)
        token_ids = torch.Tensor(token_ids).long()
        mask_ids = torch.Tensor(mask_ids).long()
        type_ids = torch.Tensor(type_ids).long()
        if torch.cuda.is_available():
            token_ids = token_ids.cuda()
            mask_ids = mask_ids.cuda()
            type_ids = type_ids.cuda()
        return token_loc, token_ids, type_ids, mask_ids

    def forward(self, texts, lens):
        token_loc, token_ids, type_ids, mask_ids = self.get_info(texts)
        bert_out = self.__encoder(token_ids, token_type_ids=type_ids, attention_mask=mask_ids)
        h = bert_out.last_hidden_state
        utt = bert_out.pooler_output
        hiddens = []
        for idx, locs in enumerate(token_loc):
            hiddens.append(h[idx][locs])
        batch = len(texts)
        max_len = max(lens)
        out = ensure_gpu(torch.zeros((batch, max_len, 768)))

        if torch.cuda.is_available():
            out = out.cuda()
        for idx, x in enumerate(hiddens):
            out[idx][:lens[idx]] = x
        return out, utt