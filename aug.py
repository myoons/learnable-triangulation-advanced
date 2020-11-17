import numpy as np
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

inputImg = img.imread("./data/human36m/processed/S1/Directions-1/imageSequence/54138969/img_000001.jpg")

augClouds = iaa.Clouds()
augFogs = iaa.Fog()
augSnowFlakes = iaa.Snowflakes(flake_size=(0.1, 0.1), speed=(0.01, 0.05))

imgClouds = augClouds.augment_image(inputImg)
imgFogs = augFogs.augment_image(inputImg)
imgSnowFlakes = augSnowFlakes.augment_image(inputImg)

fig = plt.figure()
rows = 2
cols = 2

fig.add_subplot(rows, cols, 1).imshow(inputImg)
fig.add_subplot(rows, cols, 2).imshow(imgClouds)
fig.add_subplot(rows, cols, 3).imshow(imgFogs)
fig.add_subplot(rows, cols, 4).imshow(imgSnowFlakes)

plt.imshow(imgClouds)
plt.show()


