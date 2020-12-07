import os
import cv2
import argparse
import numpy as numpy

import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

from mvn.utils import vis, cfg
from mvn.models.triangulation import AlgebraicTriangulationNet

from ccreate import finalize

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

def set_dataloader(config):

    dataset_1 = torchvision.datasets.ImageFolder(root='fatboy/camera_1',
                                            transform=transforms.Compose([
                                            transforms.Resize((384,384)),
                                            transforms.ToTensor(),
                                            ]))

    dataloader_1 = torch.utils.data.DataLoader(dataset=dataset_1,
                                                    batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
                                                    shuffle=False,
                                                    drop_last = True
                                                    )

    dataset_2 = torchvision.datasets.ImageFolder(root='fatboy/camera_2',
                                            transform=transforms.Compose([
                                            transforms.Resize((384,384)),
                                            transforms.ToTensor(),
                                            ]))

    dataloader_2 = torch.utils.data.DataLoader(dataset=dataset_2,
                                                    batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
                                                    shuffle=False,
                                                    drop_last = True
                                                    )

    dataset_4 = torchvision.datasets.ImageFolder(root='fatboy/camera_4',
                                            transform=transforms.Compose([
                                            transforms.Resize((384,384)),
                                            transforms.ToTensor(),
                                            ]))

    dataloader_4 = torch.utils.data.DataLoader(dataset=dataset_4,
                                                    batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
                                                    shuffle=False,
                                                    drop_last = True
                                                    )

    dataset_6 = torchvision.datasets.ImageFolder(root='fatboy/camera_6',
                                            transform=transforms.Compose([
                                            transforms.Resize((384,384)),
                                            transforms.ToTensor(),
                                            ]))

    dataloader_6 = torch.utils.data.DataLoader(dataset=dataset_6,
                                                batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
                                                shuffle=False,
                                                drop_last = True
                                                )

    return zip(dataloader_1, dataloader_2, dataloader_4, dataloader_6)

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
    model = AlgebraicTriangulationNet(config, device=device).to(device)

    if config.model.init_weights: # eval
        state_dict = torch.load(config.model.checkpoint) # Load Saved Model
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print("**** Successfully loaded pretrained weights for whole model; {} ****".format(type(model)))

    # datasets
    print("Loading data...")
    batch_size = config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size
    dataloaders = set_dataloader(config)
    
    frame = 0

    with torch.no_grad():

        for data_1, data_2, data_4, data_6 in dataloaders:

            images_1 = data_1[0]
            images_2 =  data_2[0]
            images_4 =  data_4[0]
            images_6 =  data_6[0]

            images_batch = torch.Tensor(4, batch_size , 3, 384, 384)

            images_batch[0] = images_1
            images_batch[1] = images_2
            images_batch[2] = images_4
            images_batch[3] = images_6

            images_batch = images_batch.permute(1, 0, 2, 3, 4).contiguous().to(device)
            proj_matricies_batch = finalize(batch_size).to(device)

            
            keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_matricies_batch)

            """
            for idx, keypoint in enumerate(keypoints_3d_pred): 

                ax = plt.axes()  
                vis.draw_3d_pose(keypoint.cpu(), ax, keypoints_mask=None, kind='human36m', radius=None, root=None, point_size=2, line_width=2, draw_connections=True)
                plt.savefig('vis_test/{}'.format(str(frame)))
                plt.close()
                
                frame += 1
                print('Frame : {}'.format(frame))
            """

            for idx, keypoint in enumerate(keypoints_2d_pred):
                for camera_idx, images in enumerate(keypoint):
                    ax = plt.axes()  
                    vis.draw_2d_pose(images.cpu(), ax, kind='human36m')
                    plt.savefig('vis_test/{}'.format(str(frame)))
                    plt.close()

                    frame += 1
                    print('Frame : {}'.format(frame))

            del keypoints_3d_pred, images_batch, keypoints_2d_pred, heatmaps_pred, confidences_pred, proj_matricies_batch
            torch.cuda.empty_cache()

if __name__ == '__main__' :

    args = parse_args()
    main(args)








