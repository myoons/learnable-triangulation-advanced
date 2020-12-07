import cv2
import numpy as np
import os
from os.path import isfile, join


pathIn= '/volume/cs409/vis_fatboy/vis_fatboy_base/'
pathOut = 'fatboy_base.avi'

fps = 40
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

#for sorting the file names properly
files = sorted(files)

for i in range(len(files)):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()