import os
import argparse
import numpy as np
import cv2

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet

from mvn.utils import img, multiview, op, vis, cfg
from mvn.datasets import human36m
from mvn.datasets import utils as dataset_utils

import matplotlib.image as img
import matplotlib.pyplot as plt

import imgaug
from imgaug import augmenters as iaa


"""
Code for image augmentations

Using Augmentations (23 Kinds)

1. Choice Standard
- Changing indoor environment to outdoor environement
- Doesn't change to position of joints

2. Choices

- Meta : Identity, ChannelShuffle
- Arithmetic : Add, AdditiveGaussianNoise, Multiply, Cutout
- Blur : GaussianBlur, AverageBlur, MotionBlur
- Color　: MultiplyAndAddToBrightness, MultiplySaturation, Grayscale, RemoveSaturation, ChangeColorTemperature
- Contrast　:　GammaContrast, HistogramEqualization
- Imgcorruptlike : Snow
- Weather : FastSnowyLandscape, Clouds, Fog, Snowflakes, Rain

"""

# Initializing Parameters
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

def aug_batch(original_batch, device):

    # original_batch : [100, 4, 3, 384, 384]
    # transposedImages : [4, 100, 384, 384, 3]
    transposedImages = original_batch.permute(1,0,3,4,2).contiguous()
    auged_batch = []

    for i in range(7):
        temp_batch=[]
        
        for idx, transposedBatch in enumerate(transposedImages):
            # transposeBatch : torch.Size([100, 384, 384, 3])       
            #transposedBatch=transposedBatch.astype(np.uint8)
            uint8Tensor = torch.as_tensor(transposedBatch, dtype=torch.uint8, device='cpu')
            #numpyBatch = uint8Tensor.numpy()
            numpyBatch = transposedBatch.cpu().contiguous().numpy()
            #numpyBatch=numpyBatch.astype(np.uint8)
            if i==0:
                temp_batch.append(iaa.ReplaceElementwise(0.1, [0, 255], per_channel=0.5).augment_images(images=numpyBatch))
            elif i==1:
                temp_batch.append(iaa.MotionBlur(k=15, angle=[-45, 45]).augment_images(images=numpyBatch))
            elif i==2:
                temp_batch.append(iaa.GaussianBlur(sigma=(0.0, 3.0)).augment_images(images=numpyBatch))
            elif i==3:
                temp_batch.append(iaa.Add((-0.4, 1.5)).augment_images(images=numpyBatch))
            elif i==4:
                temp_batch.append(iaa.Fog().augment_images(images=numpyBatch))
            elif i==5:
                temp_batch.append(iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)).augment_images(images=numpyBatch))
            elif i==6:
                temp_batch.append(iaa.Rain().augment_images(images=numpyBatch))
        temp_batch = torch.FloatTensor(temp_batch).to(device)

        temp_batch = temp_batch.permute(1, 0, 4, 2, 3).contiguous()  # temp_batch : [batchSize, 4, 3, 384, 384]
        auged_batch.append(temp_batch)

    return auged_batch


def aug_vis(model, config, dataloader, device, aug):
    
    model_type = config.model.name # alg, vol
    model.eval()

    with torch.no_grad():

        iterator = enumerate(dataloader) # train_loader / val_loader , len(dataloader) :48743
        # 한 배치당 image 100개
        for iter_i, batch in iterator: # enumerate(dataloader) , batch : Dictionary {key : images, detections, cameras, keypoints_3d, indexes}
            
            if batch is None:
                print("Found None batch")
                continue

            images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(batch, device, config)
            keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None

            print('aaaaaaaaaaaaaaaaaaaa : ', proj_matricies_batch.size())

            auged_batch_proto = aug_batch(images_batch, device) # [7, batchSize, 4, 3, 384, 384]

            augList = []
            augList.append(iaa.ReplaceElementwise(0.1, [0, 255], per_channel=0.5))
            augList.append(iaa.MotionBlur(k=15, angle=[-45, 45]))
            augList.append(iaa.GaussianBlur(sigma=(0.0, 3.0)))
            augList.append(iaa.Add((-0.4, 1.5)))
            augList.append(iaa.Fog())
            augList.append(iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)))
            augList.append(iaa.Rain())

            for bIdx, auged_batch in enumerate(auged_batch_proto): # [batchSize, 4, 3, 384, 384]
                # prediction with model (input)
                if model_type == "alg" or model_type == "ransac":
                    keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(auged_batch, proj_matricies_batch, batch)
                elif model_type == "vol":
                    keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = model(auged_batch, proj_matricies_batch, batch)

                np_confidence = confidences_pred.detach().cpu().numpy()
                
                for i in range(3):

                    keypoints_vis = vis.visualize_batch(
                        auged_batch, heatmaps_pred, keypoints_2d_pred, proj_matricies_batch,
                        keypoints_3d_gt, keypoints_3d_pred,
                        kind='human36m',
                        cuboids_batch=cuboids_pred,
                        confidences_batch=confidences_pred,
                        batch_index=i, size=5,
                        max_n_cols=10,
                        augmentation=augList[bIdx]
                    )
            
            break
        

def main(args):

    print("**** Number of available GPUs: {} ****".format(torch.cuda.device_count()))

    is_distributed = init_distributed(args)
    
    if is_distributed:
        device = torch.device(args.local_rank)
    else:
        device = torch.device(0)

    # config
    config = cfg.load_config(args.config) # config file

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

    # datasets
    print("Loading data...")

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

    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    aug = iaa.imgcorruptlike.GaussianNoise(severity=2)
    aug_vis(model, config, val_dataloader, device, aug)


if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)

