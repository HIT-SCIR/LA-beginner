from __future__ import annotations
from typing import Any, Dict, List, Tuple
from torch.utils.data import DataLoader


class Vocabulary(object):
    """词表类

    保存模型中用到的单词表
    可以看作Dict[str, int]
    对于未出现的单词会返回<UNK>

    Attributes:
        word2idx: 保存单词对应的编号
        word_number: 单词总数

    """

    def __init__(self):
        """实例化词表类

        每次实例化时，会默认添加<PAD> <UNK> <BOS>
        """

        self.word2idx: Dict[str, int] = {}
        self.word_number = 0

        self.append('<PAD>')
        self.append('<UNK>')
        self.append('<BOS>')

    @staticmethod
    def load(path: str) -> Vocabulary:
        """加载单词表

        从指定路径加载单词表
        单词表文件格式为：word idx

        Args:
            path: 单词表路径

        Returns:
            Vocabulary: 单词表实例
        """
        vocabulary = Vocabulary()
        with open(path, 'r') as f:
            for line in f:
                info = line.split()
                vocabulary.word2idx[info[0]] = int(info[1])
                vocabulary.word_number += 1
        return vocabulary

    def dump(self, path: str) -> None:
        """保存单词表

        向指定路径保存单词表
        单词表文件格式为：word idx

        Args:
            path: 保存路径
        """
        with open(path, 'w') as f:
            for word in self.word2idx:
                f.write(f'{word} {self.word2idx[word]}\n')

    def append(self, word: str) -> None:
        """向单词表添加新单词

        新单词默认编号为当前单词表长度
        若单词已在词表中，则不做处理

        Args:
            word: 新加入单词
        """
        if word in self.word2idx:
            return
        self.word2idx[word] = self.word_number
        self.word_number += 1

    # 使该类可以用[]运算符枚举
    def __getitem__(self, word: str) -> int:
        if not(word in self.word2idx):
            return self.word2idx['<UNK>']
        return self.word2idx[word]

    # 定义len()函数对该类实例的行为
    def __len__(self) -> int:
        return self.word_number


class Sentence(object):
    """句子类

    存储模型用到的句子信息

    Attributes:
        sentence: 组成句子的单词
        pos: 各个位置的词性信息
        dependent: 各个位置的依存词
    """

    def __init__(self, sentence: List[str], pos: List[str], dependent: List[int]):
        """实例化句子类

        至少有一个依存词的位置为0

        Args:
            sentence: 组成句子的单词
            pos: 各个位置的词性信息
            dependent: 各个位置的依存词
        """
        self.sentence = sentence
        self.pos = pos
        self.dependent = dependent

    # 定义hash()函数对该类实例的行为
    def __hash__(self) -> int:
        return hash(self.sentence)


class DataManager(object):
    """数据管理类

    保存所有的训练数据

    Attributes:
        sentence: 所有句子信息
        mode: train, valid或test，数据的用途
    """

    def __init__(self, mode: str):
        """实例化数据管理类

        Args:
            mode: 数据用途
        """
        self.sentences: List[Sentence] = []
        self.mode = mode

    @staticmethod
    def load(path: str, mode: str, max_line: int) -> Tuple[DataManager, Vocabulary]:
        """加载数据

        数据格式参考./data中文件

        Args:
            path: 加载路径
            mode: 数据用途
            max_line: 读入句子数量，用于debug

        Returns:
            Datamanage: 数据
            Vocabulary: 单词表
        """
        count = 0
        dm = DataManager(mode)

        # 仅对训练数据构建词表
        if mode == 'train':
            vocabulary = Vocabulary()

        with open(path, 'r') as f:
            # 在每个句子前添加<BOS>，作为树根
            sentence = Sentence(['<BOS>'], ['BOS'], [-1])
            for line in f:
                current_info = line.strip().split()
                if len(current_info) == 0:
                    count += 1
                    dm.sentences.append(sentence)
                    sentence = Sentence(['<BOS>'], ['BOS'], [-1])
                    continue
                if not(max_line is None) and count == max_line:
                    break

                current_word = current_info[1].lower()
                current_pos = current_info[4]
                current_dependent = int(current_info[6])

                sentence.sentence.append(current_word)
                sentence.pos.append(current_pos)
                sentence.dependent.append(current_dependent)

                if mode == 'train':
                    vocabulary.append(current_word)
                    vocabulary.append(current_pos)

        if mode == 'train':
            return dm, vocabulary
        return dm, None

    def sentence(self) -> List[List[str]]:
        '''返回数据中所有句子

        Returns:
            List[List[str]]: 所有句子
        '''
        return [s.sentence for s in self.sentences]

    def pos(self) -> List[List[str]]:
        '''返回数据中所有词性信息

        Returns:
            List[List[str]]: 词性信息
        '''
        return [s.pos for s in self.sentences]

    def dependent(self) -> List[List[int]]:
        '''返回数据中所有依存词

        Returns:
            List[List[str]]: 依存词
        '''
        return [s.dependent for s in self.sentences]

    def package(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        '''将数据封装

        将数据封装为可迭代形式
        具体信息可参考torch.utils.data.DataLoader

        Args:
            batch_size: 批大小
            shuffle: 每次读取DataLoader是否打乱顺序

        Returns:
            DataLoader: 封装数据
        '''
        sentence: List[List[str]] = []
        pos: List[List[str]] = []
        dependent: List[List[int]] = []
        for s in self.sentences:
            sentence.append(s.sentence)
            pos.append(s.pos)
            dependent.append(s.dependent)

        return DataLoader(dataset=_Dataset(sentence, pos, dependent),
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=_collate_fn)


# DataLoader中数据具体的组织形式
class _Dataset(object):
    def __init__(self, sentence: List[List[str]], pos: List[List[str]], dependent: List[List[int]]):
        self.sentence = sentence
        self.pos = pos
        self.dependent = dependent

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, i: int):
        return self.sentence[i], self.pos[i], self.dependent[i]


# 指导DataLoader如何返回数据
def _collate_fn(batch: _Dataset) -> List[List[Any]]:
    attr_count = len(batch[0])
    ret = [[] for _ in range(attr_count)]

    for i in range(len(batch)):
        for j in range(attr_count):
            ret[j].append(batch[i][j])

    return ret
