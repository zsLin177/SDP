# -*- coding: utf-8 -*-

import torch
from supar.utils import Config
from supar.utils.logging import init_logger, logger
from supar.utils.parallel import init_device


def parse(parser):
    parser.add_argument('--conf', '-c', help='path to config file')
    parser.add_argument('--path', '-p', help='path to model file')
    parser.add_argument('--device', '-d', default='-1', help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=16, type=int, help='max num of threads')
    parser.add_argument('--batch-size', default=5000, type=int, help='batch size')
    parser.add_argument("--local_rank", type=int, default=-1, help='node rank for distributed training')
    parser.add_argument('--min_freq', default=2, type=int, help='minimum frequency needed to include a token in the vocabulary')
    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args = Config(**vars(args))
    Parser = args.pop('Parser')

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    init_device(args.device, args.local_rank)
    init_logger(logger, f"{args.path}.{args.mode}.log")
    logger.info('\n' + str(args))

    if args.mode == 'train':
        # min_freq看看是多少
        parser = Parser.build(**args)
        args.update({
            'mu': .0,
            'nu': 0.95,
            'lr': 1e-3,
            'weight_decay': 3e-9
        })
        parser.train(**args)
    elif args.mode == 'evaluate':
        parser = Parser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = Parser.load(args.path)
        parser.predict(**args)
