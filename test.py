# coding:utf-8
import math

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pylab


label = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
onehot = tf.one_hot(label, depth=10)
print(onehot)