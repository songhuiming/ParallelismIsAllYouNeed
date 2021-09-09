
import time
import torch
import torch.distributed as dist
import apex
from apex import amp
import horovod.torch as hvd

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op = dist.ReduceOp.SUM)
    return rt / float(nprocs)

def train(train_dataloader, model, criterion, optimizer, epoch, local_rank, args):
    total_loss, total_accuracy = 0, 0
    model.train()
    starttime = time.time()
    for step, batch in enumerate(train_dataloader):
        batch = [r.cuda(local_rank, non_blocking = True) for r in batch]
        sent_id, mask, labels = batch
        predictions = model(sent_id, mask)
        loss = criterion(predictions, labels)
        accuarcy = torch.mean((torch.argmax(predictions, axis=1) == labels).float())
        if step%100==0:
            print(f"On rank {local_rank}, loss is {loss:6.5f}, acc is {accuarcy:6.5f}")
        model.zero_grad()
        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, args['nprocs'])
        reduced_accuracy = reduce_mean(accuarcy, args['nprocs'])
        total_loss = total_loss + reduced_loss.item()
        total_accuracy = total_accuracy + reduced_accuracy.item()
        # compute gradient and do SGD backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
        optimizer.step()
    if local_rank == 0:
        avg_loss = total_loss / len(train_dataloader)
        avg_accuracy = total_accuracy / len(train_dataloader)
        return avg_loss, avg_accuracy

def validate(val_dataloader, model, criterion, local_rank, args):
    total_loss, total_accuracy = 0, 0
    model.eval()
    with torch.no_grad():
        end = time.time()
        for step, batch in enumerate(val_dataloader):
            batch = [r.cuda(local_rank, non_blocking=True) for r in batch]
            sent_id, mask, labels = batch
            predictions = model(sent_id, mask)
            loss = criterion(predictions, labels)
            accuarcy = torch.mean((torch.argmax(predictions, axis=1) == labels).float())
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args['nprocs'])
            reduced_accuracy = reduce_mean(accuarcy, args['nprocs'])
            total_loss = total_loss + reduced_loss.item()
            total_accuracy = total_accuracy + reduced_accuracy.item()
    if local_rank == 0:
        avg_loss = total_loss / len(val_dataloader)
        avg_accuracy = total_accuracy / len(val_dataloader)
        return avg_loss, avg_accuracy

def save_bestmodel(state, is_best, outpath, filename = "bestmodel.weights.pt"):
    if is_best:
        torch.save(state, outpath + filename)

def apex_train(train_dataloader, model, criterion, optimizer, epoch, local_rank, args):
    total_loss, total_accuracy = 0, 0
    model.train()
    starttime = time.time()
    for step, batch in enumerate(train_dataloader):
        batch = [r.cuda(local_rank, non_blocking = True) for r in batch]
        sent_id, mask, labels = batch
        predictions = model(sent_id, mask)
        loss = criterion(predictions, labels)
        accuarcy = torch.mean((torch.argmax(predictions, axis=1) == labels).float())
        if step%100==0:
            print(f"On rank {local_rank}, loss is {loss:6.5f}, acc is {accuarcy:6.5f}")
        model.zero_grad()
        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, args['nprocs'])
        reduced_accuracy = reduce_mean(accuarcy, args['nprocs'])
        total_loss = total_loss + reduced_loss.item()
        total_accuracy = total_accuracy + reduced_accuracy.item()
        # compute gradient and do SGD backpropagation
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
        optimizer.step()
    if local_rank == 0:
        avg_loss = total_loss / len(train_dataloader)
        avg_accuracy = total_accuracy / len(train_dataloader)
        return avg_loss, avg_accuracy

def hvd_train(train_dataloader, model, criterion, optimizer, epoch, local_rank, args):
    total_loss, total_accuracy = 0, 0
    model.train()
    starttime = time.time()
    for step, batch in enumerate(train_dataloader):
        batch = [r.cuda(local_rank, non_blocking = True) for r in batch]
        sent_id, mask, labels = batch
        predictions = model(sent_id, mask)
        loss = criterion(predictions, labels)
        accuarcy = torch.mean((torch.argmax(predictions, axis=1) == labels).float())
        if step%300==0:
            print(f"On rank {hvd.local_rank()}={local_rank}, loss is {loss:6.5f}, acc is {accuarcy:6.5f}")
        model.zero_grad()
        reduced_loss = hvd.allreduce(loss)
        reduced_accuracy = hvd.allreduce(accuarcy)
        total_loss = total_loss + reduced_loss.item()
        total_accuracy = total_accuracy + reduced_accuracy.item()
        # compute gradient and do SGD backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
        optimizer.step()
    avg_loss = total_loss / len(train_dataloader)
    avg_accuracy = total_accuracy / len(train_dataloader)
    if local_rank == 0:
        return avg_loss, avg_accuracy

def hvd_validate(val_dataloader, model, criterion, local_rank, args):
    total_loss, total_accuracy = 0, 0
    model.eval()
    with torch.no_grad():
        end = time.time()
        for step, batch in enumerate(val_dataloader):
            batch = [r.cuda(local_rank, non_blocking=True) for r in batch]
            sent_id, mask, labels = batch
            predictions = model(sent_id, mask)
            loss = criterion(predictions, labels)
            accuarcy = torch.mean((torch.argmax(predictions, axis=1) == labels).float())
            reduced_loss = hvd.allreduce(loss)
            reduced_accuracy = hvd.allreduce(accuarcy)
            total_loss = total_loss + reduced_loss.item()
            total_accuracy = total_accuracy + reduced_accuracy.item()
    avg_loss = total_loss / len(val_dataloader)
    avg_accuracy = total_accuracy / len(val_dataloader)
    if local_rank == 0:
        return avg_loss, avg_accuracy
