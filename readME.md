# Parallelism is All You Need

When BERT [1] model was introduced in 2018, it has about 110 million parameters for base and 345 million for large model. Cross lingual model XLM-R [2] has 550M parameters. GPT-2-xl [3] has 1.5 billion parameters while GPT-3 [4] has about 175 billion parameters. Text-To-Text Transfer Transformer (T5) [5] has 11 billion parameters. As the model becomes bigger and bigger, it brings new challenges: it takes more and more time to train or fine tune the model; some models are too big to be loaded into one GPU because of limited GPU memory; or even if it can be loaded into GPU, the batch size has to be set as very small since the parameters and gradients need to be saved for parameter update and it will result in lacking of GPU memory.

![pic1](https://github.com/songhuiming/ParallelismIsAllYouNeed/blob/main/figure/Picture1.png)

![pic2](https://github.com/songhuiming/ParallelismIsAllYouNeed/blob/main/figure/Picture2.png)



## 1. [By pytorch `multiprocess.spawn`](https://github.com/songhuiming/ParallelismIsAllYouNeed/blob/main/1.multiprocessing.distributed.py)
pytorch `multiprocess.spawn` will create the processes and pass the local_rank as well as the arguments to it. You can run python from command line to initialize one process and then spawn multiple process inside the code.

It is as simple as these 3 steps:
1. initial the process by `dist.init_process_group`
2. send model to `torch.nn.parallel.DistributedDataParallel`
3. distribute the data by `torch.utils.data.distributed.DistributedSampler`

```python
mp.spawn(main_worker, nprocs = args['nprocs'], args = (args['nprocs'], args))
```

Initialize the device and set model to each device:
```python
    dist.init_process_group(backend = 'nccl', init_method="tcp://127.0.0.1:23456", world_size=args['nprocs'], rank=local_rank)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    ...
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [local_rank])
```

Set up the sampler to distribute the data:
```python
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
```

Need to make sure all the loss is calcualted on all devices by
```python
torch.distributed.barrier()
```

## 2. [pytorch distributed launcher](https://github.com/songhuiming/ParallelismIsAllYouNeed/blob/main/2.distributed.py)
`torch.distributed.launch` can run the python code distributely. It is similar to the spawn above but it will launch n processes immediately when the command is kicked off.

It has the similar 3 steps as above.

```python
def main():
    ...
    main_worker(args['local_rank'], args['nprocs'], args)

def main_worker(local_rank, nprocs, args):
    ...
    dist.init_process_group(backend = 'nccl', init_method="tcp://127.0.0.1:23456", world_size=args['nprocs'], rank=local_rank)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    args['batch_size'] = int(args['batch_size'] / args['nprocs'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [local_rank])
    ...
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
```


## 3. [nvidia apex mixed precision](https://github.com/songhuiming/ParallelismIsAllYouNeed/blob/main/3.apex.distributed.py)
apex will not only provide parallel training, but also automatically help to manage the precision of the parameters in the models, optimizers and loss to reduce the GPU memory usage and improve the training speed.

It includes these key steps:
1. use apex to initialize the model and optimizer so that it can automatically manage the precision
2. send the model to `apex.parallel.DistributedDataParallel(model)`
3. scale the loss

```python
    from apex import amp
    ...
    dist.init_process_group(backend = 'nccl', init_method="tcp://127.0.0.1:23456", world_size=args['nprocs'], rank=local_rank)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    args['batch_size'] = int(args['batch_size'] / args['nprocs'])
    ...
    model, optimizer = amp.initialize(model, optimizer)
    model = apex.parallel.DistributedDataParallel(model)
    ...
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
```

## 4. [hovorod](https://github.com/songhuiming/ParallelismIsAllYouNeed/blob/main/4.horovod.distributed.py)
hovorod do the parallel computing through MPI ring AllReduce which can reduce the data trandfer and thus improve the training speed.

It mainly includes these steps:
1. init the process `hvd.init()`
2. broadcast the model parameters `hvd.broadcast_parameters(model.state_dict(), root_rank=0)`
3. broadcast the optimizer `hvd.broadcast_optimizer_state(optimizer, root_rank=0)`

```python

def main():
    ...
    hvd.init()
    args['local_rank'] = hvd.local_rank()
    ...
    main_worker(args['local_rank'], args['nprocs'], args)

def main_worker(local_rank, nprocs, args):
    ...
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    ...
    optimizer = torch.optim.AdamW(model.parameters(), args['learning_rate'], weight_decay=args['weight_decay'])
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=hvd.Compression.fp16, op=hvd.Average)
```

## 5. [huggingface accelerate](https://github.com/songhuiming/ParallelismIsAllYouNeed/blob/main/5.hgf.accelerate.py)
Huggingface Accelerate is a wrapper for PyTorch to make the parallel training much easier.

It mainly needs these steps:
1. send the model, optimizer, and data loader to accelerator.
2. backpropagate with accelerator

```python
    accelerator = Accelerator(fp16=args['fp16'], cpu=args['cpu'])
    model = model.to(accelerator.device)

    optimizer= AdamW(params = model.parameters(), lr = lr, correct_bias = correct_bias)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader)

    ...
    accelerator.backward(loss)
```

## 6. [gradients accumulate]()
1. split the mini-batch data into n steps defined in the accumulation steps
2. for each split, 
```python
    for step, batch in enumerate(train_dataloader):
        split_size = args['train_batch_size'] / args['gradient_accumulation_steps']
        split_accum = zip(*[torch.split(x, int(split_size)) for x in batch])
        for j, batch_split in enumerate(split_accum):
            batch_split = [r.to(device) for r in batch_split]
            sent_id, mask, labels = batch_split
            # reset gradients tensors
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
```
