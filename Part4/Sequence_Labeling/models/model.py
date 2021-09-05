from models.module import LSTMEncoder, BertEncoder
import torch.nn as nn
import torch.nn.functional as F


class SequenceLabelingModel(nn.Module):
    """
    Sequence labeling model using BiLSTM.
    """

    def __init__(self, args, word_num, slot_num):
        super(SequenceLabelingModel, self).__init__()
        self.dropout_rate = args.dropout_rate

        self.embedding = nn.Embedding(word_num, args.word_embedding_dim)
        self.bi_lstm = LSTMEncoder(args.word_embedding_dim, args.encoder_hidden_dim, self.dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(args.encoder_hidden_dim, args.encoder_hidden_dim // 2),
            nn.LeakyReLU(args.alpha),
            nn.Linear(args.encoder_hidden_dim // 2, slot_num)
        )

    def forward(self, text, seq_lens):
        """Forward process for sequence tagger

        :param text: padded indexed text (batch_size * max_sent_len)
        :param seq_lens: is the length of original input text.
        :return: log_softmax result for each position (batch_size * max_sent_len * slot_num)
        """

        text_embedding = F.dropout(self.embedding(text), p=self.dropout_rate)
        lstm_hidden = F.dropout(self.bi_lstm(text_embedding, seq_lens))
        slot_logits = self.classifier(lstm_hidden)
        return F.log_softmax(slot_logits, dim=-1)


class SequenceLabelingModelWithBert(nn.Module):
    '''
    Sequence labeling with Bert
    '''

    def __init__(self, args, slot_num):
        super(SequenceLabelingModelWithBert, self).__init__()
        self.dropout_rate = args.dropout_rate

        self.bert = BertEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(768, 768 // 2),
            nn.LeakyReLU(args.alpha),
            nn.Linear(768 // 2, slot_num)
        )

    def forward(self, text, seq_lens):
        """Forward process for sequence tagger

        :param text: padded indexed text (batch_size * max_sent_len)
        :param seq_lens: is the length of original input text.
        :return: log_softmax result for each position (batch_size * max_sent_len * slot_num)
        """
        bert_hidden, bert_cls = self.bert(text, seq_lens)
        bert_hidden = F.dropout(bert_hidden)
        slot_logits = self.classifier(bert_hidden)
        return F.log_softmax(slot_logits, dim=-1)