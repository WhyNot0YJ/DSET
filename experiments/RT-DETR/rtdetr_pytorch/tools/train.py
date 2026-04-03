"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig, parse_cli
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = parse_cli(args.update)
    update_dict.update({
        k: v for k, v in args.__dict__.items()
        if k not in ['config', 'resume', 'tuning', 'amp', 'seed', 'test_only', 'update'] and v is not None
    })

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning,
        **update_dict
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')
    args = parser.parse_args()

    main(args)
