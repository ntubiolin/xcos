import os
import json
import argparse
import pickle
from copy import copy

import torch

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer.trainer import Trainer
from utils.logger import Logger
from utils.logging_config import logger
import global_variables


def get_instance(module, name, config, *args, **kargs):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'], **kargs)


def main(config, args):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    if 'valid_data_loaders' in config.keys():
        valid_data_loaders = [
            getattr(module_data, entry['type'])(**entry['args'])
            for entry in config['valid_data_loaders']
        ]
    else:
        valid_data_loaders = [data_loader.split_validation()]

    # build model architecture
    model = get_instance(
        module_arch, 'arch', config,
    )
    model.summary()

    # setup instances of losses
    losses = {
        entry.get('nickname', entry['type']): (
            getattr(module_loss, entry['type'])(**entry['args']),
            entry['weight']
        )
        for entry in config['losses']
    }

    # setup instances of metrics
    metrics = [
        getattr(module_metric, entry['type'])(**entry['args'])
        for entry in config['metrics']
    ]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = Trainer(model, losses, metrics, optimizer,
                      resume=args.resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loaders=valid_data_loaders,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger,
                      **config['trainer_args'])

    if args.pretrained is not None:
        trainer.load_pretrained(args.pretrained)

    if args.mode == 'test':
        # this line is to solve the error described in https://github.com/pytorch/pytorch/issues/973
        torch.multiprocessing.set_sharing_strategy('file_system')
        saved_keys = ['verb_logits', 'noun_logits', 'uid', 'verb_class', 'noun_class']
        for loader in trainer.valid_data_loaders:
            file_path = os.path.join(args.save_dir, loader.name + '.pkl')
            if os.path.exists(file_path) and args.skip_exists:
                logger.warning(f'Skipping inference and saving {file_path}')
                continue
            inference_results = trainer.inference(loader, saved_keys)
            with open(file_path, 'wb') as f:
                logger.info(f'Saving results on loader {loader.name} into {file_path}')
                pickle.dump(inference_results, f)
    else:
        trainer.train()


def extend_config(config, config_B):
    new_config = copy(config)
    for key, value in config_B.items():
        if key in new_config.keys():
            if key == 'name':
                value = f"{new_config[key]}_{value}"
            else:
                logger.warning(f"Overriding '{key}' in config")
            del new_config[key]
        new_config[key] = value
    return new_config


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--configs', default=None, type=str, nargs='+',
                        help=('Configuraion files. Note that the duplicated entries of later files will',
                              ' overwrite the former ones.'))
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                        help='path to pretrained checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--save_dir', default=None, type=str, help='Path to save the inference results.')
    parser.add_argument('--skip_exists', action='store_true', help='Skip inference when saving files already exist.')
    args = parser.parse_args()

    assert args.resume is not None or args.configs is not None, 'At least one of resume or configs should be provided.'
    if args.mode == 'test':
        # assertions to make sure user do provide a valid path
        assert args.save_dir is not None
        if os.path.exists(args.save_dir):
            logger.warning(f'The directory {args.save_dir} already exists.')
        else:
            os.makedirs(args.save_dir)
    return args


if __name__ == '__main__':
    args = parse_args()
    config = {}
    if args.resume:
        # load config file from checkpoint, this will include the training information (epoch, optimizer parameters)
        config = torch.load(args.resume)['config']
    if args.configs:
        # load config files, the overlapped entries will be overwriten
        for config_file in args.configs:
            config = extend_config(config, json.load(open(config_file)))

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    global_variables.global_config = config.get('global_config', {})
    logger.info(f'Experiment name: {config["name"]}')
    main(config, args)
