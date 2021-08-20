import os
import torch
import numpy
import random
from config import args
from utils.data import DataManager
from model import Parser
from processor import Processor


if __name__ == '__main__':
    # 设置随机数种子，保证实验结果可重复
    torch.random.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    numpy.random.seed(args.random_seed)

    # 模型训练部分
    if args.mode == 'train':
        # 加载数据
        train_data, vocabulary = DataManager.load(
            args.train_data_path, 'train', args.max_line)
        valid_data, _ = DataManager.load(
            args.valid_data_path, 'valid', args.max_line)
        # 保存单词表
        vocabulary.dump(f'{args.model_path}/vocabulary.txt')

        # 实例化模型
        # 若模型已存在，则加载后继续训练
        if os.path.exists(f'{args.model_path}/{args.decoder_type}.pkl'):
            model = torch.load(f'{args.model_path}/{args.decoder_type}.pkl')
        else:
            model = Parser(len(vocabulary), args.embedding_dim, args.hidden_size,
                        args.layer_num, args.dropout_rate, vocabulary['<PAD>'])

        # 实例化处理类
        processor: Processor = Processor(args.batch_size, vocabulary, model)
        processor.fit(f'{args.model_path}/{args.decoder_type}.pkl',
                      args.epoch,
                      args.learning_rate,
                      train_data,
                      valid_data)
    # 预测部分
    elif args.mode == 'test':
        # 处理过程与训练部分相似
        # 预测结果保存在result中
        test_data, _ = DataManager.load(
            args.predict_data_path, 'test', args.max_line)
        processor = Processor.load(args.model_path, args.batch_size)
        result, acc = processor.predict_and_evaluate(test_data)
        print(f'evaluate on test data: {acc}')
