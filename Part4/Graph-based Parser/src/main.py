import os
import torch
import numpy
import random
from config import args
from utils.data import DataManager, Vocabulary
from utils.model import Parser
from utils.processor import Processor


if __name__ == '__main__':
    torch.random.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    numpy.random.seed(args.random_seed)

    if args.mode == 'train':
        train_data, vocabulary = DataManager.load(
            args.train_data_path, 'train', args.max_line)
        valid_data: DataManager = DataManager.load(
            args.valid_data_path, 'valid', args.max_line)
        train_data: DataManager
        vocabulary: Vocabulary
        vocabulary.dump(f'{args.model_path}/vocabulary.txt')

        if os.path.exists(f'{args.model_path}/{args.decoder_type}.pkl'):
            model = torch.load(f'{args.model_path}/{args.decoder_type}.pkl')
        else:
            model = Parser(len(vocabulary), args.embedding_dim, args.hidden_size,
                        args.layer_num, args.dropout_rate, vocabulary['<PAD>'])

        processor: Processor = Processor(args.batch_size, vocabulary, model)
        processor.fit(f'{args.model_path}/{args.decoder_type}.pkl',
                      args.epoch,
                      args.learning_rate,
                      train_data,
                      valid_data)
    elif args.mode == 'test':
        test_data = DataManager.load(
            args.predict_data_path, 'test', args.max_line)
        processor = Processor.load(args.model_path, args.batch_size)
        result, acc = processor.predict_and_evaluate(test_data)
        print(f'evaluate on test data: {acc}')
