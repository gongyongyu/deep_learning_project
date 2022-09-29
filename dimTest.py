# coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
X = tf.constant([2013, 2014, 2015, 2016, 2017])
Y = tf.constant([12000, 14000, 15000, 16500, 17500])

dataset = tf.data.Dataset.from_tensor_slices((X, Y))

for x, y in dataset:
    print(x.numpy(), y.numpy())
"""

(train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data(
    'C:\\Users\\gongyy\\Desktop\\DNN\\deep_learning_project\\data_source\\mnist.npz')

train_data = np.expand_dims(train_data.astype(np.float32) / 255., axis=-1)
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))

mnist_dataset = mnist_dataset.prefetch(buffer_size=tf.experimental.AUTOTUNE)


def rot90():
    pass


mnist_dataset = mnist_dataset.map(map_func=rot90, num_parallel_calls=2)
