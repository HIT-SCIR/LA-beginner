# -*- coding: utf-8 -*-#

import os
import numpy as np
from copy import deepcopy
from collections import Counter

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Alphabet(object):
    """
    Storage and serialization a set of elements.
    """

    def __init__(self, name, if_use_pad, if_use_unk):

        self.__name = name
        self.__if_use_pad = if_use_pad
        self.__if_use_unk = if_use_unk

        self.__index2instance = []
        self.__instance2index = {}

        # Counter Object record the frequency
        # of element occurs in raw text.
        self.__counter = Counter()

        if if_use_pad:
            self.__sign_pad = "<PAD>"
            self.add_instance(self.__sign_pad)
        if if_use_unk:
            self.__sign_unk = "<UNK>"
            self.add_instance(self.__sign_unk)

    @property
    def name(self):
        return self.__name

    def add_instance(self, instance):
        """ Add instances to alphabet.

        1, We support any iterative data structure which
        contains elements of str type.

        2, We will count added instances that will influence
        the serialization of unknown instance.

        :param instance: is given instance or a list of it.
        """

        if isinstance(instance, (list, tuple)):
            for element in instance:
                self.add_instance(element)
            return

        # We only support elements of str type.
        assert isinstance(instance, str)
        # count the frequency of instances.
        self.__counter[instance] += 1

        if instance not in self.__instance2index:
            self.__instance2index[instance] = len(self.__index2instance)
            self.__index2instance.append(instance)

    def get_index(self, instance):
        """ Serialize given instance and return.

        For unknown words, the return index of alphabet
        depends on variable self.__use_unk:

            1, If True, then return the index of "<UNK>";
            2, If False, then return the index of the
            element that hold max frequency in training data.

        :param instance: is given instance or a list of it.
        :return: is the serialization of query instance.
        """

        if isinstance(instance, (list, tuple)):
            return [self.get_index(elem) for elem in instance]

        assert isinstance(instance, str)

        try:
            return self.__instance2index[instance]
        except KeyError:
            if self.__if_use_unk:
                return self.__instance2index[self.__sign_unk]
            else:
                max_freq_item = self.__counter.most_common(1)[0][0]
                return self.__instance2index[max_freq_item]

    def get_instance(self, index):
        """ Get corresponding instance of query index.

        if index is invalid, then throws exception.

        :param index: is query index, possibly iterable.
        :return: is corresponding instance.
        """

        if isinstance(index, list):
            return [self.get_instance(elem) for elem in index]

        return self.__index2instance[index]

    def __len__(self):
        return len(self.__index2instance)

    def __str__(self):
        return 'Alphabet {} contains about {} words: \n\t{}'.format(self.name, len(self), self.__index2instance)


class TorchDataset(Dataset):
    """
    Helper class implementing torch.utils.data.Dataset to
    instantiate DataLoader which deliveries data batch.
    """

    def __init__(self, text, slot):
        self.__text = text
        self.__slot = slot

    def __getitem__(self, index):
        return self.__text[index], self.__slot[index]

    def __len__(self):
        # Pre-check to avoid bug.
        assert len(self.__text) == len(self.__slot)
        return len(self.__text)


class DatasetManager(object):

    def __init__(self, args):

        # Instantiate alphabet objects.
        self.__word_alphabet = Alphabet('word', if_use_pad=True, if_use_unk=True)
        self.__slot_alphabet = Alphabet('slot', if_use_pad=False, if_use_unk=False)

        # Record the raw text of dataset.
        self.__text_word_data = {}
        self.__text_slot_data = {}

        # Record the serialization of dataset.
        self.__digit_word_data = {}
        self.__digit_slot_data = {}

        self.__args = args

    @property
    def test_sentence(self):
        return deepcopy(self.__text_word_data['test'])

    @property
    def word_alphabet(self):
        return deepcopy(self.__word_alphabet)

    @property
    def slot_alphabet(self):
        return deepcopy(self.__slot_alphabet)

    @property
    def num_epoch(self):
        return self.__args.num_epoch

    @property
    def batch_size(self):
        return self.__args.batch_size

    @property
    def learning_rate(self):
        return self.__args.learning_rate

    @property
    def l2_penalty(self):
        return self.__args.l2_penalty

    @property
    def save_dir(self):
        return self.__args.save_dir

    @property
    def slot_forcing_rate(self):
        return self.__args.slot_forcing_rate

    def show_summary(self):
        """
        :return: show summary of dataset, training parameters.
        """

        print("Training parameters are listed as follows:\n")

        print('\tnumber of train sample:                    {};'.format(len(self.__text_word_data['train'])))
        print('\tnumber of dev sample:                      {};'.format(len(self.__text_word_data['dev'])))
        print('\tnumber of test sample:                     {};'.format(len(self.__text_word_data['test'])))
        print('\tnumber of epoch:						    {};'.format(self.num_epoch))
        print('\tbatch size:							    {};'.format(self.batch_size))
        print('\tlearning rate:							    {};'.format(self.learning_rate))
        print('\trandom seed:							    {};'.format(self.__args.random_state))
        print('\trate of l2 penalty:					    {};'.format(self.l2_penalty))
        print('\trate of dropout in network:                {};'.format(self.__args.dropout_rate))

        print("\nEnd of parameters show. Save dir: {}.\n\n".format(self.save_dir))

    def quick_build(self):
        """
        Convenient function to instantiate a dataset object.
        """

        train_path = os.path.join(self.__args.data_dir, 'train.txt')
        dev_path = os.path.join(self.__args.data_dir, 'dev.txt')
        test_path = os.path.join(self.__args.data_dir, 'test.txt')

        self.add_file(train_path, 'train', if_train_file=True)
        self.add_file(dev_path, 'dev', if_train_file=False)
        self.add_file(test_path, 'test', if_train_file=False)

        os.makedirs(self.save_dir, exist_ok=True)

    def add_file(self, file_path, data_name, if_train_file):
        text, slot = self.__read_file(file_path)

        if if_train_file:
            self.__word_alphabet.add_instance(text)
        self.__slot_alphabet.add_instance(slot)

        # Record the raw text of dataset.
        self.__text_word_data[data_name] = text
        self.__text_slot_data[data_name] = slot

        # Serialize raw text and stored it.
        self.__digit_word_data[data_name] = self.__word_alphabet.get_index(text)
        if if_train_file:
            self.__digit_slot_data[data_name] = self.__slot_alphabet.get_index(slot)

    @staticmethod
    def __read_file(file_path):
        """ Read data file of given path.

        :param file_path: path of data file.
        :return: list of sentence, list of slot.
        """

        texts, slots = [], []
        with open(file_path, 'r', encoding="utf8") as fr:
            sentences = fr.read().strip().split('\n\n')
            for sentence in sentences:
                text, slot = [], []
                tokens = sentence.split('\n')
                for token in tokens[:-1]:
                    t, s = token.split()
                    text.append(t)
                    slot.append(s)
                texts.append(text)
                slots.append(slot)

        return texts, slots

    def get_data_loader(self, data_name, batch_size=None, is_digital=True, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size

        if is_digital:
            text = self.__digit_word_data[data_name]
            slot = self.__digit_slot_data[data_name]
        else:
            text = self.__text_word_data[data_name]
            slot = self.__text_slot_data[data_name]
        dataset = TorchDataset(text, slot)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.__collate_fn)

    def get_bert_data_loader(self, data_name, batch_size=None, is_digital=True, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size
        text = self.__text_word_data[data_name]
        if is_digital:
            slot = self.__digit_slot_data[data_name]
        else:
            slot = self.__text_slot_data[data_name]
        dataset = TorchDataset(text, slot)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.__collate_fn)

    @staticmethod
    def add_padding(texts, slots, digital=True):
        """pad the texts to the max length and sort the texts by length
        for LSTM pack_padded_sequence

        :param texts: list of list of text
        :param slots: list of list of slot labels (if required)
        :param digital: padding 0 or '<PAD>'
        :return: list of sorted padded texts, list of sorted slots, list of seq lens
        """
        len_list = [len(text) for text in texts]
        max_len = max(len_list)

        # Get sorted index of len_list.
        sorted_index = np.argsort(len_list)[::-1]

        trans_texts, seq_lens, trans_slots = [], [], []

        for index in sorted_index:
            seq_lens.append(deepcopy(len_list[index]))
            trans_texts.append(deepcopy(texts[index]))
            trans_slots.append(deepcopy(slots[index]))
            if digital:
                trans_texts[-1].extend([0] * (max_len - len_list[index]))
            else:
                trans_texts[-1].extend(['<PAD>'] * (max_len - len_list[index]))

        return trans_texts, trans_slots, seq_lens

    @staticmethod
    def __collate_fn(batch):
        """
        helper function to instantiate a DataLoader Object.
        """

        n_entity = len(batch[0])
        modified_batch = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(batch)):
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch
