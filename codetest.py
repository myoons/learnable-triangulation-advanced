import torch
import numpy as np
import os
from itertools import islice 
from datetime import datetime

bboxes = np.load('./data/human36m/extra/bboxes-Human36M-GT.npy', allow_pickle=True).item()

print(bboxes.keys(),'\n')
print(bboxes['S1'].keys(),'\n')
print(bboxes['S1']['Eating-2'].keys(),'\n')
print(bboxes['S1']['Eating-2']['55011271'].shape,'\n') # (2721,4) : Frame이 2721개 
print(bboxes['S1']['Eating-2']['55011271']) # S1 의 Eating-2 pose의 55011271 카메라의 bboxes이다.

def square_the_bbox(bbox):
    # bbox example : [202 556 594 695]
    top, left, bottom, right = bbox
    width = right - left
    height = bottom - top

    if height < width:
        center = (top + bottom) * 0.5
        top = int(round(center - width * 0.5))
        bottom = top + width
    else:
        center = (left + right) * 0.5
        left = int(round(center - height * 0.5))
        right = left + height

    print('top : {} \t left : {} \t bottom : {} \t right : {}'.format(top, left, bottom, right))
    return top, left, bottom, right

for subject in bboxes.keys(): # S1
    for action in bboxes[subject].keys(): # Action 
        for camera, bbox_array in bboxes[subject][action].items(): # Camera, bbox_array = (2721, 4) array
            for frame_idx, bbox in enumerate(bbox_array):
                bbox[:] = square_the_bbox(bbox)
