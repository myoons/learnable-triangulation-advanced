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

from random import *

import imgaug
from imgaug import augmenters as iaa

from scipy.ndimage.filters import gaussian_filter1d
import cv2

all_loss=[]
epoch_count=1

def select_intensity(idx):

    if idx == 0:
        value = randint(-20, 20)
        return iaa.Add(value=value)
    
    elif idx == 1:
        mul = uniform(0.8, 1.2)
        return iaa.Multiply(mul=mul)
    
    elif idx == 2:
        sigma = uniform(0.0, 3.0)
        return iaa.GaussianBlur(sigma=sigma)
    
    elif idx == 3:
        k = randint(3, 7)
        angle= randint(0, 360)
        direction = uniform(-1.0, 1.0)
        order=1
        return iaa.MotionBlur(k=k, angle=angle, direction=direction, order=order)

    elif idx == 4:
        alpha = uniform(0.0, 1.0)
        return iaa.Grayscale(alpha=alpha)
    
    elif idx == 5:
        gain = randint(3, 10)
        cutoff = uniform(0.4, 0.6)
        return iaa.SigmoidContrast(gain=gain, cutoff=cutoff)
    
    elif idx == 6:
        gain = uniform(0.6, 1.4)
        return iaa.LogContrast(gain=gain)
    
    elif idx == 7:
        alpha = uniform(0.0, 1.0)
        lightness = uniform(0.75, 2.0)
        return iaa.Sharpen(alpha=alpha, lightness=lightness)
    
    elif idx == 8:
        mul = uniform(-3.0, 3.0)
        return iaa.color.MultiplyHue(mul=mul)
    
    elif idx == 9:
        value = randint(-255, 255)
        return iaa.color.AddToHue(value=value)

    elif idx == 10:
        return iaa.arithmetic.Invert()

    elif idx == 11:
        return iaa.Identity()


def aug_batch(original_batch, device):

    # original_batch : [8, 4, 3, 384, 384]
    # transposedImages : [8, 4, 384, 384, 3]
    transposedImages = original_batch.permute(0,1,3,4,2)
            
    for batch_idx, transposedBatch in enumerate(transposedImages): # [4, 384, 384, 3]
        
        numpy_batch = cv2.normalize(transposedBatch.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) # [4, 384, 384, 3]
        
        # augmentation 고정
        selected_augmentation = randint(0,11) # 11가지 augmentation
        aug = select_intensity(selected_augmentation)

        # augmented image
        result = aug.augment_images(images=numpy_batch) 

        original_batch[batch_idx] = torch.tensor(np.array(result, dtype='float32') / 255).permute(0, 3, 1, 2)

    return torch.tensor(original_batch)

# Initializing Parameters
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored") 
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")
    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--logdir", type=str, default="/Vol1/dbstore/datasets/k.iskakov/logs/multi-view-net-repr", help="Path, where logs will be stored")

    args = parser.parse_args()
    return args

# Set train_loader, validation_loader, train_sampler
def setup_human36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None

    if is_train:

        # train
        train_dataset = human36m.Human36MMultiViewDataset(
            h36m_root=config.dataset.train.h36m_root, # ./data/human36m/processed/
            pred_results_path=config.dataset.train.pred_results_path if hasattr(config.dataset.train, "pred_results_path") else None, # None
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256), # [384, 384]
            labels_path=config.dataset.train.labels_path, # ./data/human36m/extra/human36m-multiview-labels-GTbboxes.npy
            with_damaged_actions=config.dataset.train.with_damaged_actions, # true
            scale_bbox=config.dataset.train.scale_bbox, # 1.0
            kind=config.kind, # human36m
            undistort_images=config.dataset.train.undistort_images, # true
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [], # []
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True, # True
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size, # 8
            shuffle=config.dataset.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views, # false
                                                     min_n_views=config.dataset.train.min_n_views, # 10 (Default)
                                                     max_n_views=config.dataset.train.max_n_views), # 31 (Default)
            num_workers=config.dataset.train.num_workers, # 8
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    # val
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

    return train_dataloader, val_dataloader, train_sampler

# Set dataloaders : Call setup_human36m_dataloaders function (iff use human36m dataset)
def setup_dataloaders(config, is_train=True, distributed_train=False):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler = setup_human36m_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader, train_sampler

# Setup directories and environments for experiment
def setup_experiment(config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title # Ex) human36m_alg_"model_name" / eval_human36m_alg_"model_name"

    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name)) # Experiment Name

    experiment_dir = os.path.join(args.logdir, experiment_name) # ./logs/"experiment_name" 디렉토리 생성
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints") # ./logs/"experiment_name"/checkpoints 디렉토리 생성
    os.makedirs(checkpoints_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer

# Code for One Epoch
def one_epoch(model, criterion, opt, config, dataloader, device, epoch, n_iters_total=0, is_train=True, caption='', master=True, experiment_dir=None, writer=None, fmatch=False):
    global epoch_count
    epoch_start_time = time.time()

    epoch_loss=[]
    # name = "train/val"
    name = "train" if is_train else "val"
    model_type = config.model.name # alg, vol

    if is_train:
        model.train()
    else:
        model.eval()

    metric_dict = defaultdict(list) # defaultdict : if there is no key, make new key:value pair with emtpy list
    results = defaultdict(list)

    # used to turn on/off gradients ;; train : enable_grad() / val : no_grad()
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad

    with grad_context():
        end = time.time()

        iterator = enumerate(dataloader) # train_loader / val_loader , len(dataloader) :48743 / 8
        if is_train and config.opt.n_iters_per_epoch is not None: # 1250
            # islice(iterable, start, stop, step) isslice(iterator, 1250) --> index 0 ~ 1249 (1249개) 할당함
            iterator = islice(iterator, config.opt.n_iters_per_epoch)

        if epoch < 10 : 
            fmatch = False
        else :
            fmatch = True

        for iter_i, batch in iterator: # enumerate(dataloader)

            # measure data loading time
            data_time = time.time() - end

            if batch is None:
                print("Found None batch")
                continue
            
            # batch['cameras']
            # [8, 4, ~, ~, 3]
            
            images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(batch, device, config)
            keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None

            if fmatch :
                images_batch = aug_batch(images_batch, device).to(device) # Currently CPU [7, 8, 4, 3, 384, 384]
                keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_matricies_batch, batch)
                batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(images_batch.shape[3:])

            else :
                keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_matricies_batch, batch)
                batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(images_batch.shape[3:])

            n_joints = keypoints_3d_pred.shape[1]

            # If GT validity is bigger than Zero : 1 ? : 0 
            keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)

            # Default : scale_keypoints_3d = 0.1 --> 오차가 1/100로 줄을듯?
            scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0

            # 1-view case (Not our case ; Muli-view)
            if n_views == 1:
                if config.kind == "human36m":
                    base_joint = 6
                elif config.kind == "coco":
                    base_joint = 11

                keypoints_3d_gt_transformed = keypoints_3d_gt.clone()
                keypoints_3d_gt_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_gt_transformed[:, base_joint:base_joint + 1]
                keypoints_3d_gt = keypoints_3d_gt_transformed

                keypoints_3d_pred_transformed = keypoints_3d_pred.clone()
                keypoints_3d_pred_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_pred_transformed[:, base_joint:base_joint + 1]
                keypoints_3d_pred = keypoints_3d_pred_transformed

            # calculate loss
            total_loss = 0.0
            loss = criterion(keypoints_3d_pred * scale_keypoints_3d, keypoints_3d_gt * scale_keypoints_3d, keypoints_3d_binary_validity_gt)

            # loss 는 X, Y, Z 오차들의 합(에 몇 숫자 조작을 한 것)
            total_loss += loss
            # metric_dict["MSESmooth"] 에 loss 값을 추가해준다. (list)
            metric_dict[f'{config.opt.criterion}'].append(loss.item()) 

            if (iter_i % 250 == 0) :
                print('--------------------------------------------------\n')
                print('Epoch :',epoch,'\t Iter :',iter_i,'\t loss :',total_loss)
                print('\n--------------------------------------------------')

            # volumetric ce loss
            use_volumetric_ce_loss = config.opt.use_volumetric_ce_loss if hasattr(config.opt, "use_volumetric_ce_loss") else False
            if use_volumetric_ce_loss: # Default : False for alg
                volumetric_ce_criterion = VolumetricCELoss()

                loss = volumetric_ce_criterion(coord_volumes_pred, volumes_pred, keypoints_3d_gt, keypoints_3d_binary_validity_gt)
                metric_dict['volumetric_ce_loss'].append(loss.item())

                weight = config.opt.volumetric_ce_loss_weight if hasattr(config.opt, "volumetric_ce_loss_weight") else 1.0
                total_loss += weight * loss
            
            metric_dict['total_loss'].append(total_loss.item())
            
            if is_train: # If in Trainloop
                opt.zero_grad() # input
                total_loss.backward() # back propagation

                # Default False
                if hasattr(config.opt, "grad_clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.opt.grad_clip / config.opt.lr)

                # metric_dict['grad_norm_times_lr'].append(config.opt.lr * misc.calc_gradient_norm(filter(lambda x: x[1].requires_grad, model.named_parameters())))

                opt.step()

            # calculate metrics
            l2 = KeypointsL2Loss()(keypoints_3d_pred * scale_keypoints_3d, keypoints_3d_gt * scale_keypoints_3d, keypoints_3d_binary_validity_gt)
            metric_dict['l2'].append(l2.item())

            # save answers for evalulation
            if not is_train:
                results['keypoints_3d'].append(keypoints_3d_pred.detach().cpu().numpy())
                results['indexes'].append(batch['indexes'])
            
            # plot visualization
            if master: # Base True --> if master : experiment result
                if n_iters_total % config.vis_freq == 0:# or total_l2.item() > 500.0:
                    print('****Plotting / n_iters_total : {} ****'.format(n_iters_total))
                    vis_kind = config.kind #human36m
                    if (config.transfer_cmu_to_human36m if hasattr(config, "transfer_cmu_to_human36m") else False):
                        vis_kind = "coco"
                    
                    for batch_i in range(min(batch_size, config.vis_n_elements)):
                        keypoints_vis = vis.visualize_batch(
                            images_batch, heatmaps_pred, keypoints_2d_pred, proj_matricies_batch,
                            keypoints_3d_gt, keypoints_3d_pred,
                            kind=vis_kind,
                            cuboids_batch=cuboids_pred,
                            confidences_batch=confidences_pred,
                            batch_index=batch_i, size=5,
                            max_n_cols=10
                        )

                        writer.add_image(f"{name}/keypoints_vis/{batch_i}", keypoints_vis.transpose(2, 0, 1), global_step=n_iters_total)
                        
                        heatmaps_vis = vis.visualize_heatmaps(
                            images_batch, heatmaps_pred,
                            kind=vis_kind,
                            batch_index=batch_i, size=5,
                            max_n_rows=10, max_n_cols=10
                        )
                        
                        writer.add_image(f"{name}/heatmaps/{batch_i}", heatmaps_vis.transpose(2, 0, 1), global_step=n_iters_total)
                        
                        if model_type == "vol":
                            volumes_vis = vis.visualize_volumes(
                                images_batch, volumes_pred, proj_matricies_batch,
                                kind=vis_kind,
                                cuboids_batch=cuboids_pred,
                                batch_index=batch_i, size=5,
                                max_n_rows=1, max_n_cols=16
                            )
                            writer.add_image(f"{name}/volumes/{batch_i}", volumes_vis.transpose(2, 0, 1), global_step=n_iters_total)

                # dump weights to tensoboard
                if n_iters_total % config.vis_freq == 0:
                    print('**** dump weights to tensoboard ****')
                    for p_name, p in model.named_parameters():
                        try:
                            writer.add_histogram(p_name, p.clone().cpu().data.numpy(), n_iters_total)
                        except ValueError as e:
                            print(e)
                            print(p_name, p)
                            exit()

                # dump to tensorboard per-iter loss/metric stats
                for title, value in metric_dict.items():
                    writer.add_scalar(f"{name}/{title}", value[-1], n_iters_total)

                # measure elapsed time
                batch_time = time.time() - end
                end = time.time()

                # dump to tensorboard per-iter time stats
                writer.add_scalar(f"{name}/batch_time", batch_time, n_iters_total)
                writer.add_scalar(f"{name}/data_time", data_time, n_iters_total)

                # dump to tensorboard per-iter stats about sizes
                writer.add_scalar(f"{name}/batch_size", batch_size, n_iters_total)
                writer.add_scalar(f"{name}/n_views", n_views, n_iters_total)

                n_iters_total += 1
                
    # calculate evaluation metrics
    if master:
        if not is_train:
            results['keypoints_3d'] = np.concatenate(results['keypoints_3d'], axis=0)
            results['indexes'] = np.concatenate(results['indexes'])

            try:
                scalar_metric, full_metric = dataloader.dataset.evaluate(results['keypoints_3d'])
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                scalar_metric, full_metric = 0.0, {}

            metric_dict['dataset_metric'].append(scalar_metric)

            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

            # dump results
            with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                pickle.dump(results, fout)

            # dump full metric
            with open(os.path.join(checkpoint_dir, "metric.json".format(epoch)), 'w') as fout:
                json.dump(full_metric, fout, indent=4, sort_keys=True)

        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)

    epoch_finish_time = time.time()

    print( '\n Epoch : {} Finished \t Duration : {} \n'.format(epoch, epoch_finish_time-epoch_start_time))
    return n_iters_total


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:

        if "WORLD_SIZE" in os.environ:
            print("**** Not Distributed , WORLD_SIZE : {} ****".format(int(os.environ["WORLD_SIZE"])))
        else:
            print("**** Not Distributed , No WORLD_SIZE ****")

        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True


def main(args):
    print("**** Number of available GPUs: {} ****".format(torch.cuda.device_count()))

    is_distributed = init_distributed(args)
    master = True

    print("****Master : {}****".format(master))

    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0

    if is_distributed:
        device = torch.device(args.local_rank)
    else:
        device = torch.device(0)

    # config
    config = cfg.load_config(args.config) # config file
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size # 1250
    print("**** n_iters_per_epoch : {} **** ".format(config.opt.n_iters_per_epoch)) 

    # Select model based on parameters
    model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet
    }[config.model.name](config, device=device).to(device)

    if config.model.init_weights: # eval
        state_dict = torch.load(config.model.checkpoint) # Load Saved Model
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print("**** Successfully loaded pretrained weights for whole model; {} ****".format(type(model)))

    # criterion; Default : MSESmooth
    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]

    if config.opt.criterion == "MSESmooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold) # Default : 400
    else:
        criterion = criterion_class()

    # optimizer; Default : Adam, lr = 1e-5
    opt = None
    if not args.eval:
        if config.model.name == "vol":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr}
                ],
                lr=config.opt.lr
            )
        else:
            # There are some parameters that shouldn't be optimized
            opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)


    # datasets
    print("Loading data...")

    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master: # If set, write experiment; 진짜 결과 낼 때 필요
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    if not args.eval: # train loop
        n_iters_total_train, n_iters_total_val = 0, 0
        
        print('----------------------------------------------------------------------------------------------\n')
        print("Start Training, \t Total Epoch :",config.opt.n_epochs,"\t Batch Size :",config.opt.batch_size, '\n')

        for epoch in range(config.opt.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            print('*******************************************************************************************************\n')
            print("Start Epoch : {} \t is_train : True \t n_iters_total_train : {} \t  n_iters_total_val : {}".format(epoch, n_iters_total_train, n_iters_total_val))

            n_iters_total_train = one_epoch(model, criterion, opt, config, train_dataloader, device, epoch, n_iters_total=n_iters_total_train, is_train=True, master=master, experiment_dir=experiment_dir, writer=writer)
            # n_iters_total_val = one_epoch(model, criterion, opt, config, val_dataloader, device, epoch, n_iters_total=n_iters_total_val, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

            if master:
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))

            print(f"{n_iters_total_train} iters done.")

            print("Finish Epoch : {} \t is_train : True \t n_iters_total_train : {} \t  n_iters_total_val : {}".format(epoch, n_iters_total_train, n_iters_total_val))
            print('\n*******************************************************************************************************')
            
            if master:
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))

            print(f"{n_iters_total_train} iters done.")

    else: # Validation Loop

        print('----------------------------------------------------------------------------------------------\n')
        print("Start Evaluating")

        if args.eval_dataset == 'train':
            # (model, criterion, opt, config, dataloader, device, epoch, n_iters_total=0, is_train=True, caption='', master=False, experiment_dir=None, writer=None)
            n_iters_total_train = one_epoch(model, criterion, opt, config, train_dataloader, device, epoch=0, n_iters_total=0, is_train=True, master=master, experiment_dir=experiment_dir, writer=writer)
        else: # Default
            n_iters_total_train = one_epoch(model, criterion, opt, config, val_dataloader, device, epoch=0, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

    print("""
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                        Everything Done
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """)

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
