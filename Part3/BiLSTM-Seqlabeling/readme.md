# Pytorch: Bi-LSTM Sequence-Labeling

## Introduction
    A simple sequence labeling with bi-lstm.
    ATIS data is used and included.

## Requirement
    - Python 3.5+
    - Pytorch 0.3+
    - pytorchnlp

## Tricks Included
    - glove embedding
    - mini-batch
    - learning rate decay
    - grad clip
    - dropout
    - sgd & adam

## Usage
    Train and run the model in the project dir.
    python3 postagger.py  --train_and_test --train_path ./data/atis_train --dev_path ./data/atis_dev --test_path ./data/atis.test.txt --label_set_path ./data/atis_slot_names.txt --model ./model/ --output ./result/result.txt --script ./eval/conlleval.pl --gpu 0  --load_data_type conll