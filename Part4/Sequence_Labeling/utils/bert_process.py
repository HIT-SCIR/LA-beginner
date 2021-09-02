# -*- coding: utf-8 -*-#

import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import numpy as np
from tqdm import tqdm


class Processor(object):
    """
    Train process manager
    """
    def __init__(self, dataset, model, args):
        """

        :param dataset: DatasetManager
        :param model: model
        :param args: hyper parameter
        """
        self.__dataset = dataset
        self.__model = model
        self.args = args
        self.__batch_size = args.batch_size

        if args.gpu:
            time_start = time.time()
            self.__model = self.__model.cuda()

            time_con = time.time() - time_start
            print("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        self.__criterion = nn.NLLLoss()

        self.__optimizer = optim.Adam(
            self.__model.parameters(), lr=self.__dataset.learning_rate, weight_decay=self.__dataset.l2_penalty
        )

    def ensure_gpu(self, tensor):
        if self.args.gpu:
            return tensor.cuda()
        return tensor

    def train(self):
        best_dev_sent = 0.0
        best_dev_slot = 0.0
        best_epoch = 0
        no_improve = 0
        dataloader = self.__dataset.get_bert_data_loader('train')
        for epoch in range(0, self.__dataset.num_epoch):
            total_slot_loss = 0.0
            time_start = time.time()
            self.__model.train()

            for text_batch, slot_batch in tqdm(dataloader, ncols=50):
                flat_slot = list(Evaluator.expand_list(slot_batch))
                slot_var = torch.LongTensor(flat_slot)
                seq_lens = [len(x) for x in text_batch]

                if self.args.gpu:
                    slot_var = slot_var.cuda()

                slot_out = self.__model(text_batch, seq_lens)
                slot_out = torch.cat([slot_out[i][:seq_lens[i]] for i in range(0, len(seq_lens))], dim=0)

                slot_loss = self.__criterion(slot_out, slot_var)

                self.__optimizer.zero_grad()
                slot_loss.backward()
                self.__optimizer.step()

                total_slot_loss += slot_loss.cpu().item()

            time_con = time.time() - time_start
            print(
                '[Epoch {:2d}]: The total slot loss on train data is {:2.6f}, cost '\
                'about {:2.6} seconds.'.format(epoch, total_slot_loss, time_con))

            change, time_start = False, time.time()
            dev_slot_f1_score = self.estimate(
                type='dev',
                test_batch=self.__batch_size)

            if  dev_slot_f1_score >= best_dev_slot:
                no_improve = 0
                best_epoch = epoch
                best_dev_slot = dev_slot_f1_score
                test_slot_f1 = self.estimate(
                    type='test', test_batch=self.__batch_size)

                print('\nTest result: epoch: {}, slot f1 score: {:.6f}.'.
                      format(epoch, test_slot_f1))

                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                os.makedirs(model_save_dir, exist_ok=True)
                torch.save(self.__model, os.path.join(model_save_dir, "model.pkl"))
                torch.save(self.__dataset, os.path.join(model_save_dir, 'dataset.pkl'))

                time_con = time.time() - time_start
                print('[Epoch {:2d}]: In validation process, the slot f1 score is {:2.6f}, '
                      'cost about {:2.6f} seconds.\n'.format(
                    epoch, dev_slot_f1_score, time_con))

            else:
                no_improve += 1

            if self.args.early_stop:
                if no_improve > self.args.patience:
                    print('early stop at epoch {}'.format(epoch))
                    break
        print('Best epoch is {}'.format(best_epoch))
        return best_epoch

    def estimate(self, type='dev', test_batch=100):
        """
        Estimate the performance of model on dev or test dataset.
        """

        if type == 'dev':
            ss, pred_slot, real_slot = self.prediction(
                self.__model, self.__dataset, "dev", test_batch, self.args)
        else:
            ss, pred_slot, real_slot = self.prediction(
                self.__model, self.__dataset, "test", test_batch, self.args)

        # using perl
        slot_f1_score = Evaluator.computeF1Score(ss, real_slot, pred_slot, os.path.join(self.args.save_dir, 'eval.txt'))
        print("slot f1: {}".format(slot_f1_score))
        return slot_f1_score

    @staticmethod
    def validate(model_path, dataset, batch_size, args):
        """
        validation will write mistaken samples to files and make scores.
        """

        if args.gpu:
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))

        ss, pred_slot, real_slot = Processor.prediction(
            model, dataset, "test", batch_size, args)

        slot_f1_score = Evaluator.computeF1Score(ss, real_slot, pred_slot, os.path.join(args.save_dir, 'eval.txt'))
        print("slot f1: {}".format(slot_f1_score))

        return slot_f1_score

    @staticmethod
    def prediction(model, dataset, mode, batch_size, args):
        model.eval()

        if mode == "dev":
            dataloader = dataset.get_data_loader('dev', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = dataset.get_data_loader('test', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        pred_slot, real_slot = [], []
        all_token = []
        with torch.no_grad():
            for text_batch, slot_batch in tqdm(dataloader, ncols=50):
                real_slot.extend(slot_batch)
                seq_lens = [len(x) for x in text_batch]
                all_token.extend(text_batch)

                slot_out = model(text_batch, seq_lens)
                slot_out = torch.cat([slot_out[i][:seq_lens[i]] for i in range(0, len(seq_lens))], dim=0).cpu().numpy()
                slot_id = np.argmax(slot_out, axis=-1).tolist()
                nested_slot = Evaluator.nested_list(slot_id, seq_lens)
                pred_slot.extend(dataset.slot_alphabet.get_instance(nested_slot))

        return all_token, pred_slot, real_slot


class Evaluator(object):

    @staticmethod
    def computeF1Score(ss, real_slots, pred_slots, file_path):
        with open(file_path, "w", encoding="utf8") as writer:
            for correct_slot, pred_slot, tokens in zip(real_slots, pred_slots, ss):
                for c, p, token in zip(correct_slot, pred_slot, tokens):
                    writer.write("{}\t{}\t{}\t{}\t{}\n".format(token, "n", "O", c, p))
        out = os.popen(
            'perl ./utils/conlleval.pl -d \"\\t\" < {}'.format(file_path)).readlines()
        f1 = float(out[1][out[1].find("FB1:") + 4:-1].replace(" ", "")) / 100
        return f1

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        trans_items = []

        count = 0
        for jdx in range(0, len(seq_lens)):
            trans_items.append(items[count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
