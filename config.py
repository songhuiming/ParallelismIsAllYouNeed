
import argparse

def config():
    args = {
        'data_dir': '/path-to-training-validation-dataset/',
        'model_type': 'bert',
        'model_name': 'bert-base-uncased',
        'model_path': '/path-to-bert-download-files/bert-base-uncased/',
        'output_dir': '/path-to-output-folder/',
        'max_seq_length': 50,
        'train_batch_size': 256,
        'gradient_accumulation_steps': 1,
        'num_train_epochs': 60,
        'weight_decay': 0.0001,
        'learning_rate': 1e-5,
        'lr_scheduler': True,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.06,
        'warmup_steps': 0,
        'max_grad_norm': 1.0,
        'notes': 'Kaggle Toxic01 data',
    }

    args['seed'] = 2018
    args['freeze_params'] = False
    args['batch_size'] = 512
    args['gradient_accumulation_steps'] = 8
    args['num_train_epochs'] = 50
    args['learning_rate'] = 1e-5
    args['num_workers'] = 0

    parser = argparse.ArgumentParser(description='PyTorch Parallelism Training')
    parser.add_argument('--distname',default='dist',help='distribution name to save the output')
    parser.add_argument('--local_rank',default=-1,type=int,help='node(GPU) rank for ddp')

    args1 = parser.parse_args()
    args.update(vars(args1))

    return args
