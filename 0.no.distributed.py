"""
pip install torch transformers accelerate datasets
python 0.no.distributed.py > /efs-storage/kaggleToxicOutput/0.log
"""

import sys, os
import argparse
import torch, time
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (AdamW, get_linear_schedule_with_warmup, set_seed)

# from datasets import load_dataset, load_metric
from utils_load_data_model import load_data, load_model, data_loader, seed_all
from model_arch import ModelArch
from config import config


args = config()

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

def training_funciton(args):
    lr = args['lr']
    num_epochs = int(args['num_epochs'])
    correct_bias = args['correct_bias']
    seed = int(args["seed"])
    batch_size = int(args["batch_size"])

    # 1. load tokenizer, model
    tokenizer, model = load_model(args)
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
    device = "cuda:0"
    model = model.to(device)

    optimizer= AdamW(params = model.parameters(), lr = lr, correct_bias = correct_bias)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps=100,
        num_training_steps = len(train_dataloader)* num_epochs,
    )

    criterion = nn.NLLLoss()
    epoch_time = []
    train_losses, valid_loeses = [], []
    for epoch in range(num_epochs):
        starttime = time.time()
        train_loss, valid_loss = [], []
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch
            outputs = model(sent_id, mask)
            loss = criterion(outputs, labels)
            accuarcy = torch.mean((torch.argmax(outputs, axis=1) == labels).float())
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        endtime = time.time()
        epoch_time.append(endtime - starttime)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch
            with torch.no_grad():
                outputs = model(sent_id, mask)
                accuarcy = torch.mean((torch.argmax(outputs, axis=1) == labels).float())
                loss = criterion(outputs, labels)
                valid_loss.append(loss.item())
        train_losses.append(train_loss)
        valid_loeses.append(valid_loss)
    print(epoch_time)

def main():
    args['fp16'] = False
    args['cpu'] = False
    config = {'lr': 2e-5, 'num_epochs':50, "correct_bias":True, 'seed':2018, 'batch_size':128, 'model_type':'bert',
              'model_path':'/path-to-bert-download-files/bert-base-uncased/'}
    args.update(config)
    training_funciton(args)

if __name__ == '__main__':
    main()


