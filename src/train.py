import os
import argparse

import torch

from utils.logging_config import logger
from pipeline import TrainingPipeline, TestingPipeline


def main(args):

    #######################
    # Setup global config #
    #######################
    from utils.global_config import global_config
    # load config file from checkpoint, this will include the training information (epoch, optimizer parameters)
    if args.resume is not None:
        logger.info("Resuming checkpoint: {} ...".format(args.resume))
        resumed_checkpoint = torch.load(args.resume)
    else:
        resumed_checkpoint = None
    setattr(args, 'resumed_checkpoint', resumed_checkpoint)

    global_config.setup(args.template_config, args.specified_configs, resumed_checkpoint)
    global_config.print_changed()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    logger.info(f'Experiment name: {global_config["name"]}')

    ##################
    # Setup pipeline #
    ##################
    if args.mode == 'train':
        pipeline = TrainingPipeline(args)
    elif args.mode == 'test':
        pipeline = TestingPipeline(args)
    else:
        raise NotImplementedError(f'Mode {args.mode} not defined.')

    ################
    # Run pipeline #
    ################

    pipeline.run()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument(
        '-tc', '--template_config', default='configs/template.json', type=str,
        help=('Template configuraion file. It should contain all default configuration '
              'and will be overwritten by specified config.')
    )
    parser.add_argument(
        '-sc', '--specified_configs', default=None, type=str, nargs='+',
        help=('Specified configuraion files. They serve as experiemnt controls and will '
              'overwrite template configs.')
    )
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

    # assert args.resume is not None or args.configs is not None, 'At least one of resume or configs should be provided'
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
    main(args)
