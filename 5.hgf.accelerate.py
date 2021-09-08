
"""
# pip install torch transformers accelerate datasets

python -m torch.distributed.launch --nproc_per_node 4 --use_env 5.hgf.accelerate.py
"""

import argparse
import os, sys
import torch, time
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (AdamW, get_linear_schedule_with_warmup, set_seed)

from accelerate import Accelerator, DistributedType
from utils_load_data_model import load_data, load_model, data_loader, seed_all, save_obj
from model_arch import ModelArch
from config import config
args = config()


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

def training_funciton(args):
    # accelarator = Accelerator(fp16=args['fp16'], cpu = args['cpu'])
    accelerator = Accelerator(fp16=args['fp16'], cpu=args['cpu'])
    lr = args['lr']
    num_epochs = int(args['num_epochs'])
    correct_bias = args['correct_bias']
    seed = int(args["seed"])
    batch_size = int(args["batch_size"])

    # 1. load tokenizer, model
    tokenizer, model = load_model(args, trfmnew='Yes')
    # 2. load_data
    train_text, train_labels, val_text, val_labels, test_text, test_labels = load_data(args)
    train_data, valid_data, test_data = data_loader(tokenizer, args)

    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   num_workers=2, pin_memory=True,
                                                   sampler=train_sampler)
    valid_sampler = torch.utils.data.RandomSampler(valid_data)
    eval_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=EVAL_BATCH_SIZE,
                                                 num_workers=2, pin_memory=True,
                                                 sampler=valid_sampler)

    set_seed(seed)

    model = model.to(accelerator.device)

    optimizer= AdamW(params = model.parameters(), lr = lr, correct_bias = correct_bias)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps=100,
        num_training_steps = len(train_dataloader)* num_epochs,
    )

    criterion = nn.NLLLoss()
    epoch_time = []
    train_losses = []
    valid_loeses = []
    for epoch in range(num_epochs):
        train_loss, valid_loss = [], []
        starttime = time.time()
        model.train()
        for step, batch in enumerate(train_dataloader):
            sent_id, mask, labels = batch
            sent_id.to(accelerator.device)
            mask.to(accelerator.device)
            labels.to(accelerator.device)
            outputs = model(sent_id, mask)
            loss = criterion(outputs, labels)
            train_loss.append(loss.item())
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # batch.to(accelerator.device)
            sent_id, mask, labels = batch
            sent_id.to(accelerator.device)
            mask.to(accelerator.device)
            labels.to(accelerator.device)
            with torch.no_grad():
                outputs = model(sent_id, mask)
                loss = criterion(outputs, labels)
                valid_loss.append(loss.item())

        endtime = time.time()
        epoch_time.append(endtime - starttime)
        train_losses.append(train_loss)
        valid_loeses.append(valid_loss)

def main():
    args['fp16'] = False
    args['cpu'] = False
    config = {'lr': 2e-5, 'num_epochs':50, "correct_bias":True, 'seed':2018, 'batch_size':128, 'model_type':'bert',
              'model_path':'/path-to-bert-download-files/bert-base-uncased/'}
    args.update(config)
    training_funciton(args)

if __name__ == '__main__':
    main()

# 'model_path':'/path-to-bert-download-files/bert-base-uncased/'


