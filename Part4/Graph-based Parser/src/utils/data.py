from __future__ import annotations
from typing import Dict, List
from torch.utils.data import DataLoader


class Vocabulary(object):
    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.word_number = 0

        self.append('<PAD>')
        self.append('<UNK>')
        self.append('<BOS>')

    @staticmethod
    def load(path: str) -> Vocabulary:
        vocabulary = Vocabulary()
        with open(path, 'r') as f:
            for line in f:
                info = line.split()
                vocabulary[info[0]] = int(info[1])
                vocabulary.word_number += 1
        return vocabulary

    def dump(self, path: str) -> None:
        with open(path, 'w') as f:
            for word in self.word2idx:
                f.write(f'{word} {self.word2idx[word]}\n')

    def append(self, word: str) -> None:
        if word in self.word2idx:
            return
        self.word2idx[word] = self.word_number
        self.word_number += 1

    def __getitem__(self, word: str) -> int:
        if not(word in self.word2idx):
            return self.word2idx['<UNK>']
        return self.word2idx[word]

    def __len__(self) -> int:
        return self.word_number


class Sentence(object):
    def __init__(self, sentence: List[str], pos: List[str], dependent: List[int]):
        self.sentence = sentence
        self.pos = pos
        self.dependent = dependent

    def __hash__(self) -> int:
        return hash(self.sentence)


class DataManager(object):
    def __init__(self, mode: str):
        self.sentences: List[Sentence] = []
        self.mode = mode

    @staticmethod
    def load(path: str, mode: str, max_line: int):
        count = 0
        dm = DataManager(mode)
        if mode == 'train':
            vocabulary = Vocabulary()

        with open(path, 'r') as f:
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
        return dm

    def sentence(self) -> List[List[str]]:
        return [s.sentence for s in self.sentences]

    def pos(self) -> List[List[str]]:
        return [s.pos for s in self.sentences]

    def dependent(self) -> List[List[int]]:
        return [s.dependent for s in self.sentences]

    def package(self, batch_size: int, shuffle: bool = True) -> DataLoader:
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


class _Dataset(object):
    def __init__(self, sentence: List[List[str]], pos: List[List[str]], dependent: List[List[int]]):
        self.sentence = sentence
        self.pos = pos
        self.dependent = dependent

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, i: int):
        return self.sentence[i], self.pos[i], self.dependent[i]


def _collate_fn(batch: _Dataset):
    attr_count = len(batch[0])
    ret = [[] for _ in range(attr_count)]

    for i in range(len(batch)):
        for j in range(attr_count):
            ret[j].append(batch[i][j])

    return ret
