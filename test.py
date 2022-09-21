# coding:utf-8
import math
import tensorflow as tf
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image

img = np.array(Image.open("data_source/5by4_image.png"))
print(img.shape)
print(img)
plt.imshow(img)
pylab.show()
