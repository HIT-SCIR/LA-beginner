# -*- coding: utf-8 -*-#
import os
import random
import numpy as np
from models.model import SequenceLabelingModel, SequenceLabelingModelWithBert
from utils.loader import DatasetManager
from utils.bert_process import Processor
from utils.config import *

if __name__ == "__main__":
    # Save training and model parameters.
    os.makedirs(args.save_dir, exist_ok=True)

    # Fix the random seed of package random.
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # Fix the random seed of Pytorch when using GPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Fix the random seed of Pytorch when using CPU.
    torch.manual_seed(args.random_state)
    torch.random.manual_seed(args.random_state)

    # Instantiate a dataset object.
    dataset = DatasetManager(args)
    dataset.quick_build()
    dataset.show_summary()

    # Instantiate a network model object.
    model = SequenceLabelingModelWithBert(
        args,
        len(dataset.slot_alphabet)
    )

    # To train and evaluate the models.
    process = Processor(dataset, model, args)
    best_epoch = process.train()
    result = Processor.validate(
        os.path.join(args.save_dir, "model/model.pkl"),
        dataset,
        args.batch_size,  args=args)
    print('\nAccepted performance: ' + str(result) + " at test dataset;\n")
