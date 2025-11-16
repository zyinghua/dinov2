# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import glob
import json
import logging
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import time
import submitit

from classifiers import LinearClassifier, AllClassifiers, create_linear_input, scale_lr, setup_linear_classifiers

logger = logging.getLogger("dit")





class PrecomputedFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, labels: torch.Tensor):
        self.features = torch.Tensor(features).contiguous()
        self.labels = torch.Tensor(labels).int().contiguous()
        self._targets = self.labels.cpu().numpy()

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

    def get_targets(self):
        return self._targets



class InfiniteDataLoader:
    def __init__(self, data_loader: torch.utils.data.DataLoader):
        self.data_loader = data_loader

    def __iter__(self):
        while True:
            for batch in self.data_loader:
                yield batch

    def __len__(self):
        return len(self.data_loader)


def get_args_parser():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )


    parser.set_defaults(
        epochs=10,
        batch_size=128,
        num_workers=8,
        epoch_length=1250,
        learning_rates=[1e-4, 1e-3, 1e-2, 1e-1],
    )
    return parser



def run_eval_linear(
    fold_idx,
    batch_size,
    epochs,
    epoch_length,
    num_workers,
    learning_rates,
    x_train = None, 
    y_train = None,
    x_val = None,
    y_val = None,
    sample_output = None,
):

    n_last_blocks_list = [1, 4]
    training_num_classes = None

    train_dataset = PrecomputedFeatureDataset(x_train, y_train)
    val_dataset = PrecomputedFeatureDataset(x_val, y_val)
    training_num_classes = len(np.unique(train_dataset.get_targets().astype(int)))
    print(f"Number of training classes: {training_num_classes}")


    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    # train_data_loader = InfiniteDataLoader(base_train_loader)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        n_last_blocks_list,
        learning_rates,
        batch_size,
        training_num_classes,
    )

    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    max_iter = epochs * epoch_length
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")

        for data, labels in train_data_loader:
            data = data.cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)
            labels_idx = labels - 1 # adjust labels to be 0-indexed
            outputs = linear_classifiers(data)

            losses = {f"loss_{k}": nn.CrossEntropyLoss()(v, labels_idx) for k, v in outputs.items()}
            loss = sum(losses.values())

            # compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # step
            optimizer.step()
            scheduler.step()

            # if epoch % 10 ==0:
            #     print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    linear_classifiers.eval()
    with torch.no_grad():
        total_correct = {k: 0 for k in linear_classifiers.classifiers_dict.keys()}
        total_samples = 0
        for data, labels in val_data_loader:
            data = data.cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)
            labels_idx = labels - 1
            outputs = linear_classifiers(data)
            batch_size = labels.size(0)
            total_samples += batch_size
            for k, v in outputs.items():
                _, preds = v.max(1)
                total_correct[k] += (preds == labels_idx).sum().item()
        for k in linear_classifiers.classifiers_dict.keys():
            acc = total_correct[k] / total_samples
            print(f"[FOLD {fold_idx}] Validation Accuracy for {k}: {acc:.4f}")

    return linear_classifiers


def main(args):

    layers = [27]
    timesteps=['0201'] #, '0101']
    imagetypes = ['clean'] #, 'gaussian']

    #################### read precomputed features and split into 4 fold  #######################
    feature_dir = "../../../scratch/linprobe_DiT_XL_2"
    features = []
    labels = []
    for layer in layers:
        for timestep in timesteps:
            for imagetype in imagetypes:
                files = sorted(glob.glob(f'{feature_dir}/ILSVRC2012_val_*_layer{layer}_t{timestep}_{imagetype}_part0000.pt'))
                # load features and labels
                features = []
                labels = []
                for idx, file in tqdm(enumerate(files)):
                    data = torch.load(file, map_location="cpu")
                    features.append(data['features'])
                    labels.append(data['labels'])
                    if idx == 0:
                        sampe_output = data['features']
                X = torch.cat(features).numpy()
                y = torch.cat(labels).numpy()
                print(f'Layer {layer}, timestep {timestep}, imagetype {imagetype}')
                skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
                    print(f"\n[FOLD {fold_idx}] Training linear model...")
                    start = time.time()
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
    #################################################################################################

                    linear_classifiers = run_eval_linear(
                        fold_idx=fold_idx,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        epoch_length=args.epoch_length,
                        num_workers=args.num_workers,
                        learning_rates=args.learning_rates,
                        x_train = X_train,
                        y_train = y_train,
                        x_val = X_val,
                        y_val = y_val,
                        sample_output = sampe_output,
                    )
                    print(f"[FOLD {fold_idx}] Time taken: {time.time() - start:.2f} seconds")

                save_dir = f"../../../layer{layer}_t{timestep}_{imagetype}_linear_classifiers/epochs_{args.epochs}"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'fold{fold_idx}.pt')
                torch.save(linear_classifiers.state_dict(), save_path)  
                print(f"Saved linear classifiers to {save_path}")
    return 0


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    
        # slurm job submission
    if args.use_slurm:
        try:
            executor = submitit.AutoExecutor(folder="logs_slurm")
            executor.update_parameters(
                mem_gb=10,
                gpus_per_node=1,
                cpus_per_task= 12,
                nodes=1,
                timeout_min=9 * 60,  
                slurm_partition="gpu",
                slurm_signal_delay_s=120,
            )
            job = executor.submit(main, args)
            print(job)
        except Exception as e:
            print("Failed to submit job to slurm, running locally")
            print(e)
            main(args)

    else:
        main(args)
