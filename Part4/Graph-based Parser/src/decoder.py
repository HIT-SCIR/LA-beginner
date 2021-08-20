from typing import List


# MST
class Edge:
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w

    def __str__(self):
        return str(self.u) + str(self.v) + str(self.w)


def chuliu_decoder(score: List[List[float]], size: int, real_dependent: List[int] = None) -> List[int]:
    '''最大树形图解码器

    采用朱刘算法
    获取除真实依存树之外的最大树形图

    Args:
        score: 打分函数，有效大小为[size, size]
        size: 节点数，含根
        real_dependent: 真实依存树

    Returns:
        List[int]: 获取所需的最大树形图
    '''
    edges: List[Edge] = []
    # 构造图中所有的边
    # 删除自环
    # 删除指向根的边
    # 如果节点数大于2，则删除所有真实依存树中的边
    for i in range(size):
        for j in range(size):
            if i != j and j != 0 and ((size <= 2 or real_dependent is None) or real_dependent[j] != i):
                edges.append(Edge(i, j, score[i][j]))

    # 最晚获取的边是我们需要的边
    # 所以需要reverse
    edges_record: List[int] = chuliu(edges, size, 0)
    edges_record.reverse()

    # 构造返回结果
    source_record: List[int] = [None for _ in range(size)]
    for i in edges_record:
        if i == -1:
            continue
        e = edges[i]
        source_record[e.v] = e.u

    return source_record


def chuliu(edges: List[Edge], size: int, root: int) -> List[Edge]:
    '''朱刘算法

    Args:
        edges: 所有可用边
        size: 有效节点数
        root: 根节点编号
    '''
    previous: List[int] = [None for _ in range(size)]
    weight: List[float] = [float("-inf") for _ in range(size)]
    edges_record: List[Edge] = [-1 for _ in range(size)]

    # 查找每个节点的最大入边
    # 记录边编号，用于获取依存树
    for i in range(len(edges)):
        e = edges[i]
        if e.u != e.v and weight[e.v] < e.w:
            previous[e.v] = e.u
            weight[e.v] = e.w
            edges_record[e.v] = i

    # 查找环
    circle_count = 0
    circle_idx: List[int] = [-1 for _ in range(size)]
    visited: List[int] = [-1 for _ in range(size)]
    for i in range(size):
        if i == root:
            continue

        j = i
        # 标注每个节点能够访问到的节点
        while visited[j] == -1 and circle_idx[j] == -1 and j != root:
            visited[j] = i
            j = previous[j]
        # 存在自环，标注环编号
        if j != root and visited[j] == i:
            circle_idx[j] = circle_count
            t = previous[j]
            while t != j:
                circle_idx[t] = circle_count
                t = previous[t]
            circle_count += 1

    # 若没有环，则直接返回
    if circle_count == 0:
        return edges_record

    # 为了便于处理，单个点视作环
    for i in range(size):
        if circle_idx[i] == -1:
            circle_idx[i] = circle_count
            circle_count += 1

    # 根据环编号重新构造边集
    # 由于需要保留边编号，所以必须保证递归时边数相同
    new_edges: List[Edge] = []
    for e in edges:
        new_edges.append(Edge(circle_idx[e.u], circle_idx[e.v], e.w))
        new_edges[-1].w -= weight[e.v]
    new_edges_record = chuliu(new_edges, circle_count, circle_idx[root])

    # 按照递归的顺序记录边集
    return edges_record + new_edges_record


# Eisner
def eisner_decoder(scores: List[List[int]], size: int, gold: List[int] = None):
    '''Eisner算法解码器

    采用Eisner算法
    获取除真实依存树之外的最大树形图

    Args:
        score: 打分函数，有效大小为[size, size]
        size: 节点数，含根
        real_dependent: 真实依存树

    Returns:
        List[int]: 获取所需的最大树形图
    '''
    import numpy as np
    scores = np.array(scores)
    nr, nc = np.shape(scores)
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    # 除根节点外节点数量
    N = size - 1

    # 初始化CYK表
    # 最后一维0/1分别表示左右子树
    complete = np.zeros([N+1, N+1, 2])
    incomplete = np.zeros([N+1, N+1, 2])
    complete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)
    incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)

    incomplete[0, :, 0] -= np.inf

    # 由小到大进行遍历
    for k in range(1, N+1):
        for s in range(N-k+1):
            t = s+k

            # 首先构造incomplete
            # 左子树
            incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(
                t+1), t, 0] + scores[t, s] + (0.0 if gold is not None and gold[s] == t else 1.0)
            incomplete[s, t, 0] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
            # 右子树
            incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(
                t+1), t, 0] + scores[s, t] + (0.0 if gold is not None and gold[t] == s else 1.0)
            incomplete[s, t, 1] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

            # 然后构造complete
            # 左子树
            complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            complete[s, t, 0] = np.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            # 右子树
            complete_vals1 = incomplete[s,
                                        (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
            complete[s, t, 1] = np.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

    heads = [-1 for _ in range(N+1)]
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
