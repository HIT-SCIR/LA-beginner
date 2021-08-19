import torch
from multiprocessing import Pool
from typing import List, Dict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
    with Pool(processes=4) as pool:
        result: torch.FloatTensor = torch.tensor(0.)
        if torch.cuda.is_available():
            result = result.cuda()

        all_loss_dependent = []
        for i in range(len(real_dependent)):
            current_dependent = real_dependent[i]
            all_loss_dependent.append(pool.apply_async(
                mst, (score[i].tolist(), current_dependent, length[i], )))
        all_loss_dependent = [x.get() for x in all_loss_dependent]

        zero = torch.tensor(0.0)
        one = torch.tensor(1.0)
        if torch.cuda.is_available():
            zero = zero.cuda()
            one = one.cuda()
        for i in range(len(real_dependent)):
            loss_dependent = all_loss_dependent[i]
            real_sum: torch.FloatTensor = sum(
                [score[i][current_dependent[j]][j] for j in range(1, len(current_dependent))])
            loss_sum: torch.FloatTensor = sum(
                [score[i][loss_dependent[j]][j] for j in range(1, len(loss_dependent))])
            result += torch.max(zero, one - real_sum + loss_sum)
        return result


class Edge:
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w

    def __str__(self):
        return str(self.u) + str(self.v) + str(self.w)


def mst(score: List[List[float]], real_dependent: List[int], size: int, u_remove: int = -1, v_remove: int = -1) -> List[int]:
    '''
    visited: List[int] = [False for _ in range(size)]
    weight: List[int] = [score[0][i] for i in range(size)]
    previous: List[int] = [0 for _ in range(size)]
    source_record: List[int] = [-1 for _ in range(size)]
    '''
    edges: List[Edge] = []
    for i in range(size):
        for j in range(size):
            if i != j and j != 0 and not(i == u_remove and j == v_remove):
                edges.append(Edge(i, j, score[i][j]))

    source_record, _ = chuliu_mst(edges, size, 0)

    if size <= 2:
        return source_record

    for i in range(1, size):
        assert source_record[i] >= 0
    if sum([0 if real_dependent[i] == source_record[i] else 1 for i in range(size)]) != 0:
        return source_record

    result: List[int] = []
    best_weight = None
    for i in range(1, size):
        current_dependent = mst(score, real_dependent,
                                size, source_record[i], i)
        current_weight = sum([score[current_dependent[i]][i]
                             for i in range(len(current_dependent))])
        if best_weight is None or current_weight > best_weight:
            best_weight = current_weight
            result = current_dependent

    for i in range(1, size):
        assert result[i] >= 0
    return result


def chuliu_mst(edges: List[Edge], size: int, root: int) -> List[int]:
    previous: List[int] = [-1 for _ in range(size)]
    weight: List[float] = [float("-inf") for _ in range(size)]
    for e in edges:
        if e.u != e.v and e.v != root and weight[e.v] < e.w:
            previous[e.v] = e.u
            weight[e.v] = e.w

    for i in range(len(edges)):
        edges[i].w -= weight[e.v]

    circle_count = 0
    circle_idx: List[int] = [-1 for _ in range(size)]
    visited: List[int] = [-1 for _ in range(size)]
    for i in range(size):
        j = i
        while visited[j] != i and circle_idx[j] == -1 and j != root:
            visited[j] = i
            j = previous[j]
        if j != root and circle_idx[j] == -1:
            while circle_idx[j] != circle_count:
                circle_idx[j] = circle_count
                j = previous[j]
            circle_count += 1

    if circle_count == 0:
        return previous, edges

    for i in range(size):
        if circle_idx[i] == -1:
            circle_idx[i] = circle_count
            circle_count += 1

    new_edges: List[Edge] = []
    for e in edges:
        new_edges.append(Edge(circle_idx[e.u], circle_idx[e.v], e.w))

    _, new_edges = chuliu_mst(new_edges, circle_count, circle_idx[root])
    for i in range(len(edges)):
        if new_edges[i].w == 0:
            previous[edges[i].v] = edges[i].u

    return previous, edges


if __name__ == '__main__':
    size = 6
    score: List[List[float]] = [[0 for _ in range(size)] for _ in range(size)]
    dependent: List[int] = [i + 1 for i in range(size-1)]

    score[0][1] = 2
    score[0][3] = 3
    score[0][2] = 0
    score[2][1] = 1
    score[1][2] = 1
    score[1][3] = 2
    score[3][2] = 1
    score[2][4] = 3
    score[3][5] = 4
    score[3][4] = 3
    score[4][5] = 6
    dependent = [0, 3, 0, 3, 3]

    print(mst(score, dependent))
