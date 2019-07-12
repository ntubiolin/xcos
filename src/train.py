import os
import json
import argparse
from copy import copy

import torch

from utils.logging_config import logger
import global_variables
from pipeline.pipeline_manager import PipelineManager


def main(config, args):
    pipeline_manager = PipelineManager(args, config)
    pipeline_manager.run()


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
