'''
Make a list of files for a large dataset.

Example:
    python scripts/make_dataset_list.py -p '../datasets/mnist/*/*' -o ../datasets/mnist_list.txt

'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA
from glob import glob
import argparse

from utils.logging_config import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--pattern', type=str, required=True,
        help='The root dataset directory pattern for glob (e.g. "../datasets/*")'
    )
    parser.add_argument(
        '-o', '--output_filename', type=str, required=True,
        help='Output txt file path'
    )
    args = parser.parse_args()
    return args


def main(args):
    abs_patterns = os.path.abspath(args.pattern)
    logger.info(abs_patterns)
    paths = sorted(glob(abs_patterns))
    logger.info(f"There are totally {len(paths)} files")
    with open(args.output_filename, 'w') as fout:
        fout.writelines([f"{p}\n" for p in paths])

    logger.info(f"{args.output_filename} written")


if __name__ == '__main__':
    args = parse_args()
    main(args)
