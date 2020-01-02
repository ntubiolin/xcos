import shutil
import os
from os.path import basename, dirname, abspath
from glob import glob

import argparse


def num_checkpoints(path):
    files = glob(os.path.join(path, '*.pth'))
    return len(files)


def collect_satisfied(args):
    collected = []
    if args.exact_path is not None:
        exp_paths = sorted(glob(os.path.join(args.root_dir_ckpts, args.exact_path)))
        if len(exp_paths) > 0:
            assert len(exp_paths) == 1
            exp_path = exp_paths[0]
            if num_checkpoints(exp_path) < args.at_least:
                collected.append(exp_path)
    else:
        arch_paths = sorted(glob(os.path.join(args.root_dir_ckpts, args.pattern)))
        for arch_path in arch_paths:
            assert os.path.isdir(arch_path)
            assert os.path.basename(arch_path) != 'runs'
            exp_paths = sorted(glob(os.path.join(arch_path, '*')))
            assert all([os.path.isdir(exp_path) for exp_path in exp_paths])

            exp_paths = [
                exp_path for exp_path in exp_paths
                if num_checkpoints(exp_path) < args.at_least
            ]
            collected.extend(exp_paths)
    return collected


def ask_one_by_one(args, collected):
    for path in collected:
        exp_name = basename(dirname(path))
        exp_time = basename(path)

        runs_dir = os.path.join(args.root_dir_runs, exp_name, exp_time)
        print('\nDelete the following directories?')
        print(path)
        if os.path.exists(runs_dir):
            print(runs_dir)
        response = input('[y/N]? ')
        if response == 'y':
            shutil.rmtree(path)
            if os.path.exists(runs_dir):
                shutil.rmtree(runs_dir)
            print('Deleted.')
        else:
            print('No deletion performed.')


def clean_empty_exp(args):
    def walk_clean(root):
        for exp_path in glob(os.path.join(root, '*')):
            assert os.path.isdir(exp_path)
            if len(os.listdir(exp_path)) == 0:
                os.rmdir(exp_path)

    walk_clean(args.root_dir_ckpts)
    walk_clean(args.root_dir_runs)


def ask_all_in_once(args, collected):
    to_delete = []
    print('The following directories will be deleted.')
    for path in collected:
        exp_name = basename(dirname(path))
        exp_time = basename(path)
        to_delete.append(path)
        print(path)

        runs_dir = os.path.join(args.root_dir_runs, exp_name, exp_time)
        if os.path.exists(runs_dir):
            to_delete.append(runs_dir)
            print(runs_dir)
        print()

    response = input('[y/N]? ')
    if response == 'y':
        for path in to_delete:
            shutil.rmtree(path)
        print('Deleted')
    else:
        print('No deletion performed.')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--at_least', type=int, default=1,
    help='The number of saved checkpoints required for not being deleted.'
         'In other words, the saved files whose #checkpoints are less than this number will be deleted (Default: 1)')
parser.add_argument(
    '-p', '--pattern', type=str,
    help='Those saved files fit this re pattern will be deleted.')
parser.add_argument(
    '--root_dir_ckpts', type=str,
    default=os.path.join(dirname(dirname(abspath(__file__))), 'saved', 'ckpts'),
    help='The save directory'
)
parser.add_argument(
    '--root_dir_runs', type=str,
    default=os.path.join(dirname(dirname(abspath(__file__))), 'saved', 'runs'),
    help='The save directory'
)
parser.add_argument(
    '-all', '--all_in_once', action='store_true',
    help='Set this to disable the one-by-one comformation'
)
parser.add_argument(
    '--exact_path', type=str, default=None,
    help='Set this to a specific experiment path (exp_name/exp_time) to delete.'
)
args = parser.parse_args()
assert args.pattern is not None or args.exact_path is not None, 'A pattern or an exact exp path must be provided.'
assert args.pattern != '', 'Pattern can not be empty.'


if __name__ == '__main__':
    collected = collect_satisfied(args)
    if len(collected) == 0:
        print('No satisfied directory.')
        exit()

    if args.all_in_once:
        ask_all_in_once(args, collected)
    else:
        ask_one_by_one(args, collected)
    clean_empty_exp(args)
