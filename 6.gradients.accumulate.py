import torch
import numpy as np

# gradients accumulate + apex mixed precision training step

# no gradients accumulate training step
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

# gradients accumulate + apex mixed precision training step
def grad_accu_train(train_dataloader, model, criterion, optimizer, epoch, local_rank, args):
    model.train()
    total_loss, total_accuracy = 0, 0
    total_preds = []
    for step, batch in enumerate(train_dataloader):
        split_size = args['train_batch_size'] / args['gradient_accumulation_steps']
        split_accum = zip(*[torch.split(x, int(split_size)) for x in batch])
        for j, batch_split in enumerate(split_accum):
            batch_split = [r.to(device) for r in batch_split]
            sent_id, mask, labels = batch_split
            model.zero_grad()
            # get model predictions for the current batch_split
            predictions = model(sent_id, mask)
            # compute the loss between actual and predicted values
            loss = criterion(predictions, labels)
            # normalize the loss
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            # add on to the total loss
            total_loss = total_loss + loss.item()
            # apex.amp mixed precision
            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
            else:
                # backward pass to calculate the gradients
                loss.backward()
                # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
            # accumulative gradients
            if (j + 1) % args['gradient_accumulation_steps'] == 0:
                # update parameters
                optimizer.step()
                # reset gradients tensors
                model.zero_grad()
            preds = predictions.detach().cpu().numpy()
            total_preds.append(preds)