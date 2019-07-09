import os
import resource

# When saving model outputs of the training set, the dictionary gets too big
# such that python yeilds "Too many open files" error and halt. Adding these 2 lines
# is to add more resource sfor thig process.
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)  # NOQA
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))  # NOQA

import pickle
import json
import argparse
from copy import copy

import torch

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer.trainer import Trainer
from utils.logger import Logger


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = get_instance(module_data, 'valid_data_loader', config)

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # setup instances of losses
    losses = {
        entry['type']: (
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
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger,
                      **config['trainer_args'])
    return trainer


def override_data_setting(config, dataset_config):
    new_config = copy(config)
    for data_loader_name in ['data_loader', 'valid_data_loader', 'test_data_loader']:
        if data_loader_name in new_config.keys():
            del new_config[data_loader_name]
        if data_loader_name in dataset_config.keys():
            new_config[data_loader_name] = dataset_config[data_loader_name]
            if 'sampling_num' in config['arch']['args'].keys():
                loader_sampling_num = new_config[data_loader_name]['args']['dataset_args']['sampling_num']
                arch_sampling_num = config['arch']['args']['sampling_num']
                if loader_sampling_num != arch_sampling_num:
                    print(f"'sampling_num' in {data_loader_name} ({loader_sampling_num}) and model architecture"
                          f" (arch_sampling_num) differs, overrided with that of model architecture")
                    new_config[data_loader_name]['args']['dataset_args']['sampling_num'] = arch_sampling_num
    return new_config


def inference_and_save(train, loader, filename, saved_keys=['label', 'logits', 'video_id']):
    if not os.path.exists(filename):
        ret = trainer.inference(loader, saved_keys=saved_keys)
        with open(filename, 'wb') as f:
            pickle.dump(ret, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--dataset_config', default=None, type=str,
                        help='the dataset config json file (will '
                        'override the train/val/test data settings in config.json)')
    args = parser.parse_args()
    assert args.resume is not None

    config = torch.load(args.resume)['config']
    if args.dataset_config is not None:
        dataset_config = json.load(open(args.dataset_config))
        config = override_data_setting(config, dataset_config)

    trainer = main(config, args.resume)
    save_dir = os.path.dirname(args.resume)
    file_prefix = os.path.basename(args.resume).split('.')[0].split('-')[1]

    filename = os.path.join(save_dir, f'{file_prefix}_train_logits.pkl')
    inference_and_save(trainer, trainer.data_loader, filename)

    filename = os.path.join(save_dir, f'{file_prefix}_valid_logits.pkl')
    inference_and_save(trainer, trainer.valid_data_loader, filename)
