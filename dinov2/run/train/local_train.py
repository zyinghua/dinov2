# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
SLURMLESS training for DINOv2.
Usage: torchrun --nproc_per_node=<num_gpus> dinov2/run/train/local_train.py --config-file <config> --output-dir <output> ...[other args]
"""

import logging
import os
import sys

from dinov2.logging import setup_logging
from dinov2.train import get_args_parser as get_train_args_parser, main as train_main

import pdb; pdb.set_trace()


logger = logging.getLogger("dinov2")


def main():
    # Parse arguments (only training args, no SLURM-specific args)
    train_args_parser = get_train_args_parser(add_help=True)
    args = train_args_parser.parse_args()
    
    setup_logging()
    
    # Check if config file exists
    if not os.path.exists(args.config_file):
        logger.error(f"Configuration file does not exist: {args.config_file}")
        return 1
    
    # Ensure output directory is set
    if not args.output_dir:
        import tempfile
        args.output_dir = os.path.join(tempfile.gettempdir(), "dinov2_training")
        logger.warning(f"No output directory specified, using: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log distributed training info
    if "RANK" in os.environ:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        logger.warning("Not running in distributed mode. Use torchrun for multi-GPU training.")
    
    # Run training
    try:
        train_main(args)
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

