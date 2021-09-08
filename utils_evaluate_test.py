"""
put the test data evaluation to this file
"""


from __future__ import absolute_import, division, print_function

import random, os, time, shutil, sys, datetime, pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed

import glob
import logging

import time, os, random, json, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import (WEIGHTS_NAME, BertConfig, BertModel, BertTokenizer,
                          XLMConfig, XLMModel, XLMTokenizer,
                          XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from model_arch import ModelArch

import imp
config = imp.load_source('config', 'config.py')
args = config.config()

args['output_dir'] = args['output_dir'] + args['distname'] + '/'

print(args)



def seed_all(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'xlm': (XLMConfig, XLMModel, XLMTokenizer),
    'xlmr': (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer)
}

# 2. load_data

def load_test(args):
    print(f"  Model will be saved in {args['output_dir']}")
    if not os.path.exists(args['output_dir']):
        os.system('mkdir -p ' + args['output_dir'])
    test_path = args['data_dir'] + "test.cleaned.text.csv"
    # test_path = args['data_dir'] + "/test.hs.clean.csv"
    testdf = pd.read_csv(test_path)
    # testdf = pd.read_csv(test_path, nrows = 1000)
    test_text, test_labels = testdf.comment_text, testdf.label
    return test_text, test_labels



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


def test_data_loader(tokenizer, args):

    test_text, test_labels = load_test(args)

    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length = args['max_seq_length'],
        pad_to_max_length=True,
        truncation=True
    )

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())

    test_data = TensorDataset(test_seq, test_mask, test_y)

    return test_data, test_y

