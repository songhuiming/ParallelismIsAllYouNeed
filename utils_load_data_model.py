
import random, os, time, shutil
import json, math, glob, logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import (WEIGHTS_NAME, BertConfig, BertModel, BertTokenizer,
                          XLMConfig, XLMModel, XLMTokenizer,
                          XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from model_arch import ModelArch


MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'xlm': (XLMConfig, XLMModel, XLMTokenizer),
    'xlmr': (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer)
}

def load_data(args):
    train_path = args['data_dir'] + "train.cleaned.text.csv"
    valid_path = args['data_dir'] + "valid.cleaned.text.csv"
    test_path = args['data_dir'] + "test.cleaned.text.csv"

    traindf = pd.read_csv(train_path)
    validdf = pd.read_csv(valid_path)
    testdf = pd.read_csv(test_path)

    train_text, train_labels = traindf.comment_text, traindf.label
    val_text, val_labels = validdf.comment_text, validdf.label
    test_text, test_labels = testdf.comment_text, testdf.label
    return train_text, train_labels, val_text, val_labels, test_text, test_labels


def load_model(args):
    args = args
    model_path = args['model_path']
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
    # config
    config = config_class.from_pretrained(model_path + 'config.json')
    config.output_hidden_states = False
    # model
    hugg_model = model_class.from_pretrained(model_path, config = config)
    # tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = ModelArch(hugg_model, args)
    return tokenizer, model


def data_loader(tokenizer, args):
    train_text, train_labels, val_text, val_labels, test_text, test_labels = load_data(args)

    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length = args['max_seq_length'],
        pad_to_max_length=True,
        # padding='max_length',
        truncation=True
    )

    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length = args['max_seq_length'],
        pad_to_max_length=True,
        # padding='max_length',
        truncation=True
    )

    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length = args['max_seq_length'],
        pad_to_max_length=True,
        # padding='max_length',
        truncation=True
    )

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())

    train_data = TensorDataset(train_seq, train_mask, train_y)
    valid_data = TensorDataset(val_seq, val_mask, val_y)
    test_data = TensorDataset(test_seq, test_mask, test_y)
    return train_data, valid_data, test_data

def seed_all(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
