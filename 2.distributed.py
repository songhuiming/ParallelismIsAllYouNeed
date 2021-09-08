"""
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 2.distributed.py --distname=dist
"""

from __future__ import absolute_import, division, print_function
import random, os, time, shutil, sys
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
import torch.utils.data.distributed as tudd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import apex
from apex import amp
from apex.parallel import DistributedDataParallel
import horovod.torch as hvd

from model_arch import ModelArch
from utils_load_data_model import load_data, load_model, data_loader, seed_all
from trainvalid import train, validate, save_bestmodel
from config import config


args = config()
args['output_dir'] = args['output_dir'] + args['distname'] + '/'

if not os.path.exists(args['output_dir']):
    os.system('mkdir -p ' + args['output_dir'])


def main():
    args['nprocs'] = torch.cuda.device_count()
    main_worker(args['local_rank'], args['nprocs'], args)

def main_worker(local_rank, nprocs, args):
    if args['seed'] is not None:
        seed_all(args['seed'])

    tokenizer, model = load_model(args)

    best_acc = 0.0
    dist.init_process_group(backend = 'nccl', init_method="tcp://127.0.0.1:23456", world_size=args['nprocs'], rank=local_rank)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    args['batch_size'] = int(args['batch_size'] / args['nprocs'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [local_rank])
    criterion = nn.NLLLoss().cuda(local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), args['learning_rate'], weight_decay=args['weight_decay'])
    cudnn.benchmark = True
    train_dataset, valid_dataset, test_dataset = data_loader(tokenizer, args)
    # prepare the train data loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True, sampler=train_sampler)
    # validation data loader
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    val_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True, sampler=valid_sampler)

    train_losses = []
    valid_losses = []
    train_accuracys = []
    valid_accuracys = []
    epoch_time = []

    for epoch in range(args['num_train_epochs']):
        starttime = time.time()
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        train_output = train(train_dataloader, model, criterion, optimizer, epoch, local_rank, args)
        valid_output = validate(val_dataloader, model, criterion, local_rank, args)
        torch.distributed.barrier()

        if args['local_rank'] == 0:
            is_best = valid_output[1] > best_acc
            best_acc = max(valid_output[1], best_acc)
            # save_bestmodel(model.module.state_dict(), is_best, outpath = args['output_dir'])
            train_losses.append(train_output[0])
            valid_losses.append(valid_output[0])
            train_accuracys.append(train_output[1])
            valid_accuracys.append(valid_output[1])
            endtime = time.time()
            epoch_time.append(endtime - starttime)


if __name__ == '__main__':
    main()