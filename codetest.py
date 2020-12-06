import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle
import copy

# from numba import cuda
import numpy as np
import cv2

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets import human36m
from mvn.datasets import utils as dataset_utils

import imgaug
from imgaug import augmenters as iaa

from scipy.ndimage.filters import gaussian_filter1d
from random import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored") 
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")
    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--logdir", type=str, default="./AugedOutputs", help="Path, where logs will be stored")

    args = parser.parse_args()
    return args


if __name__ == '__main__' :

    args = parse_args()

    config = cfg.load_config(args.config) # config file

    val_dataset = human36m.Human36MMultiViewDataset(
            h36m_root=config.dataset.val.h36m_root,
            pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None, # None
            train=False,
            test=True,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256), # [384, 384]
            labels_path=config.dataset.val.labels_path, # ./data/human36m/extra/human36m-multiview-labels-GTbboxes.npy
            with_damaged_actions=config.dataset.val.with_damaged_actions, # true
            retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test, # 1
            scale_bbox=config.dataset.val.scale_bbox, # 1.0
            kind=config.kind, # human36m
            undistort_images=config.dataset.val.undistort_images, # true
            ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [], # []
            crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True, # True
        )

    val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
            shuffle=config.dataset.val.shuffle,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                        min_n_views=config.dataset.val.min_n_views,
                                                        max_n_views=config.dataset.val.max_n_views),
            num_workers=config.dataset.val.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    iterator = enumerate(val_dataloader) # train_loader / val_loader , len(dataloader) :48743
    for iter_i, batch in iterator: # enumerate(dataloader) , batch : Dictionary {key : images, detections, cameras, keypoints_3d, indexes}
                
        if batch is None:
            print("Found None batch")
        continue

        images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(batch, device, config)
        print(proj_matricies_batch.size())