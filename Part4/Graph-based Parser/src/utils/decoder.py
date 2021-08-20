from math import ceil
from random import randint
from typing import List
from config import args


# MST
class Edge:
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w

    def __str__(self):
        return str(self.u) + str(self.v) + str(self.w)


def chuliu_decoder(score: List[List[float]], size: int, real_dependent: List[int], u_remove: int = -1, v_remove: int = -1) -> List[int]:
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

    edges_record: List[int] = chuliu(edges, size, 0)
    edges_record.reverse()

    source_record: List[int] = [None for _ in range(size)]
    for i in edges_record:
        if i == -1:
            continue
        e = edges[i]
        source_record[e.v] = e.u

    if size <= 2:
        return source_record

    if sum([0 if real_dependent[i] == source_record[i] else 1 for i in range(size)]) != 0:
        return source_record

    result: List[int] = []
    best_weight = None
    edges_removed: List[int] = [randint(1, size-1)
                                for _ in range(ceil(args.random_rate * size))]
    for i in edges_removed:
        current_dependent = chuliu_decoder(score, real_dependent,
                                size, source_record[i], i)
        current_weight = sum([score[current_dependent[i]][i]
                             for i in range(len(current_dependent))])
        if best_weight is None or current_weight > best_weight:
            best_weight = current_weight
            result = current_dependent

    return result


def chuliu(edges: List[Edge], size: int, root: int):
    previous: List[int] = [None for _ in range(size)]
    weight: List[float] = [float("-inf") for _ in range(size)]
    edges_record: List[Edge] = [-1 for _ in range(size)]
    for i in range(len(edges)):
        e = edges[i]
        if e.u != e.v and weight[e.v] < e.w:
            previous[e.v] = e.u
            weight[e.v] = e.w
            edges_record[e.v] = i

    circle_count = 0
    circle_idx: List[int] = [-1 for _ in range(size)]
    visited: List[int] = [-1 for _ in range(size)]
    for i in range(size):
        if i == root:
            continue

        j = i
        while visited[j] == -1 and circle_idx[j] == -1 and j != root:
            visited[j] = i
            j = previous[j]
        if j != root and visited[j] == i:
            circle_idx[j] = circle_count
            t = previous[j]
            while t != j:
                circle_idx[t] = circle_count
                t = previous[t]
            circle_count += 1

    if circle_count == 0:
        return edges_record

    for i in range(size):
        if circle_idx[i] == -1:
            circle_idx[i] = circle_count
            circle_count += 1

    new_edges: List[Edge] = []
    for e in edges:
        new_edges.append(Edge(circle_idx[e.u], circle_idx[e.v], e.w))
        new_edges[-1].w -= weight[e.v]
    new_edges_record = chuliu(new_edges, circle_count, circle_idx[root])

    return edges_record + new_edges_record


# Eisner
def eisner_decoder(scores: List[List[int]], size: int, gold: List[int] = None):
    '''
    Parse using Eisner's algorithm.
    '''
    import numpy as np
    scores = np.array(scores)
    nr, nc = np.shape(scores)
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = size - 1  # Number of words (excluding root).

    # Initialize CKY table.
    complete = np.zeros([N+1, N+1, 2])  # s, t, direction (right=1).
    incomplete = np.zeros([N+1, N+1, 2])  # s, t, direction (right=1).
    # s, t, direction (right=1).
    complete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)
    # s, t, direction (right=1).
    incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)

    incomplete[0, :, 0] -= np.inf

    # Loop from smaller items to larger items.
    for k in range(1, N+1):
        for s in range(N-k+1):
            t = s+k

            # First, create incomplete items.
            # left tree
            incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(
                t+1), t, 0] + scores[t, s] + (0.0 if gold is not None and gold[s] == t else 1.0)
            incomplete[s, t, 0] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
            # right tree
            incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(
                t+1), t, 0] + scores[s, t] + (0.0 if gold is not None and gold[t] == s else 1.0)
            incomplete[s, t, 1] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            # left tree
            complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            complete[s, t, 0] = np.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            # right tree
            complete_vals1 = incomplete[s,
                                        (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
            complete[s, t, 1] = np.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

    value = complete[0][N][1]
    heads = [-1 for _ in range(N+1)]  # -np.ones(N+1, dtype=int)
    backtrack_eisner(incomplete_backtrack,
                     complete_backtrack, 0, N, 1, 1, heads)

    value_proj = 0.0
    for m in range(1, N+1):
        h = heads[m]
        value_proj += scores[h, m]

    return heads


def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    '''
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the 
    head of each word.
    '''
    if s == t:
        return
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == 0:
            backtrack_eisner(incomplete_backtrack,
                             complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack,
                             complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack,
                             complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack,
                             complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = incomplete_backtrack[s][t][direction]
        if direction == 0:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack,
                             complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack,
                             complete_backtrack, r+1, t, 0, 1, heads)
            return
        else:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack,
                             complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack,
                             complete_backtrack, r+1, t, 0, 1, heads)
            return


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
    dependent = [None, 0, 3, 0, 3, 3]

    print(eisner_decoder(score, 6, dependent))
