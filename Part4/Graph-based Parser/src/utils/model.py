import torch
from multiprocessing import Pool
from typing import List
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import args
from utils.decoder import chuliu_decoder, eisner_decoder


class Parser(torch.nn.Module):
    def __init__(self,
                 vocabulary_size: int,
                 embedding_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 padding_idx: int):
        super(Parser, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocabulary_size,
                                            embedding_dim=embedding_dim,
                                            padding_idx=padding_idx)
        self.lstm = torch.nn.LSTM(input_size=2 * embedding_dim,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout,
                                  bidirectional=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=4 * hidden_size,
                            out_features=hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_size,
                            out_features=1))

    def forward(self, sentence: torch.LongTensor, pos: torch.LongTensor, length: torch.LongTensor) -> torch.FloatTensor:
        # [batch_size, length, embedding_dim]
        sentence = self.embedding(sentence)
        pos = self.embedding(pos)

        # [batch_size, length, 2 * embedding_dim]
        feature = torch.cat([sentence, pos], dim=-1)
        feature = pack_padded_sequence(feature, length, True, False)

        # [batch_size, length, 2 * hidden_dim]
        feature, _ = self.lstm(feature)
        feature = pad_packed_sequence(feature, True)
        feature: torch.FloatTensor = feature[0]

        length = max(length)
        # [batch_size, length, 1, 2 * hidden_dim]
        feature_a = feature.unsqueeze(2).repeat(1, 1, length, 1)
        # [batch_size, 1, length, 2 * hidden_dim]
        feature_b = feature.unsqueeze(1).repeat(1, length, 1, 1)
        # [batch_size, length, length, 4 * hidden_dim]
        feature = torch.cat((feature_a, feature_b), -1)

        # [batch_size, length, length, 1]
        score: torch.FloatTensor = self.mlp(feature)
        # [batch_size, length, length]
        score = score.squeeze(-1)

        return score


def hinge_loss(score: torch.FloatTensor, real_dependent: List[List[int]], length: List[int]) -> torch.FloatTensor:
    if args.decoder_type == 'eisner':
        decoder = eisner_decoder
    elif args.decoder_type == 'chuliu':
        decoder = chuliu_decoder

    with Pool(processes=args.core_num) as pool:
        result: torch.FloatTensor = torch.tensor(0.)
        if torch.cuda.is_available():
            result = result.cuda()

        all_loss_dependent = []
        for i in range(len(real_dependent)):
            current_dependent = real_dependent[i]
            all_loss_dependent.append(pool.apply_async(
                decoder, (score[i].tolist(), length[i], current_dependent, )))
        all_loss_dependent = [x.get() for x in all_loss_dependent]

        zero = torch.tensor(0.0)
        one = torch.tensor(1.0)
        if torch.cuda.is_available():
            zero = zero.cuda()
            one = one.cuda()
        for i in range(len(real_dependent)):
            current_dependent = real_dependent[i]
            loss_dependent = all_loss_dependent[i]
            real_sum: torch.FloatTensor = sum(
                [score[i][current_dependent[j]][j] for j in range(1, len(current_dependent))])
            loss_sum: torch.FloatTensor = sum(
                [score[i][loss_dependent[j]][j] for j in range(1, len(loss_dependent))])
            result += torch.max(zero, one - real_sum + loss_sum)
        return result
