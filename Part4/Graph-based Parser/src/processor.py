from __future__ import annotations
import torch
from tqdm import tqdm
from typing import Tuple, List
from utils.data import DataManager, Vocabulary
from model import hinge_loss
from decoder import chuliu_decoder, eisner_decoder
from config import args


class Processor(object):
    '''处理类

    将数据与模型隔离
    用于训练、预测

    Attributes:
        batch_size: 批处理大小
        vocabulary: 单词表
        model: 模型
    '''

    def __init__(self,
                 batch_size: int,
                 vocabulary: Vocabulary,
                 model: torch.nn.Module = None):
        '''实例化处理类

        Args:
            batch_size: 批处理大小
            vocabulary: 单词表
            model: 模型
        '''
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.model = model

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def fit(self,
            save_path: str,
            epoch: int,
            lr: float,
            train_data: DataManager,
            valid_data: DataManager = None) -> None:
        '''训练

        未提供评价数据时，使用训练数据进行评价
        每次评价数据指标上升时，保存模型

        Args:
            save_path: 模型保存路径
            epoch: 训练轮数
            lr: 学习率
            train_data: 训练数据
            valid_data: 评价数据
        '''
        self.model.train()
        loss_function = hinge_loss
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        if valid_data is None:
            valid_data = train_data

        _, best_acc = self.predict_and_evaluate(valid_data)
        package = train_data.package(self.batch_size)
        if args.show_tqdm:
            package = tqdm(package)
        for e in range(epoch):
            loss_sum = 0
            for sentence, pos, dependent in package:
                # 封装数据
                packed_sentence = self._wrap_sentence(sentence)
                packed_pos = self._wrap_sentence(pos)
                length = torch.LongTensor([len(s) for s in sentence])
                if torch.cuda.is_available():
                    packed_sentence = packed_sentence.cuda()
                    packed_pos = packed_pos.cuda()

                # [batch_size, length, length]
                score = self.model(packed_sentence, packed_pos, length)
                loss = loss_function(score, dependent, length.tolist())
                loss_sum += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 每次得到更好的模型时，更新模型
            _, current_acc = self.predict_and_evaluate(valid_data)
            if current_acc > best_acc:
                torch.save(self.model, save_path)
                best_acc = current_acc

            print(
                f'epoch {e}: average loss is {loss_sum / len(train_data.sentence())}, best_acc is {best_acc}')

    def predict_and_evaluate(self, data: DataManager) -> Tuple[List[List[int]], float]:
        '''预测

        默认数据包含真实依存树

        Args:
            data: 预测数据

        Returns:
            List[List[int]]: 预测依存树
            float: 正确率
        '''
        if args.decoder_type == 'eisner':
            decoder = eisner_decoder
        elif args.decoder_type == 'chuliu':
            decoder = chuliu_decoder

        self.model.eval()
        match_count = 0
        result_dependent: List[List[str]] = []
        package = data.package(self.batch_size)
        if args.show_tqdm:
            package = tqdm(package)
        for sentence, pos, dependent in package:
            packed_sentence = self._wrap_sentence(sentence)
            packed_pos = self._wrap_sentence(pos)
            length = torch.LongTensor([len(s) for s in sentence])
            if torch.cuda.is_available():
                packed_sentence = packed_sentence.cuda()
                packed_pos = packed_pos.cuda()

            score = self.model(packed_sentence, packed_pos, length)
            current_dependent = [decoder(score[i].tolist(), length[i], None)
                                 for i in range(len(sentence))]

            # 统计正确的编号数
            for i in range(len(sentence)):
                for j in range(1, len(current_dependent[i])):
                    if dependent[i][j] == current_dependent[i][j]:
                        match_count += 1

            result_dependent = result_dependent + current_dependent

        self.model.train()
        return result_dependent, match_count / sum([len(d) for d in result_dependent])

    @ staticmethod
    def load(path: str, batch_size: int) -> Processor:
        '''加载模型信息

        Args:
            path: 模型地址
            batch_size: 批大小
        '''
        vocabulary = Vocabulary.load(f'{path}/vocabulary.txt')
        model = torch.load(f'{path}/{args.decoder_type}.pkl')
        processor = Processor(batch_size, vocabulary, model)
        return processor

    def _wrap_sentence(self, sentence: List[List[str]]) -> torch.LongTensor:
        '''封装信息

        将每个单词变为对应的编号
        对句子做padding

        Args:
            sentence: 待处理句子

        Returns:
            torch.LongTensor: 处理结果
        '''
        length = len(max(sentence, key=len))
        idxes = [[self.vocabulary[x] for x in s] for s in sentence]
        idxes = [s + [self.vocabulary['<PAD>']
                      for _ in range(length - len(s))] for s in idxes]
        return torch.LongTensor(idxes)
