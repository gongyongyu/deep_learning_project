# coding:utf-8
import math

import numpy as np
import tensorflow as tf
import argparse


class SimpleNet(tf.keras.Model):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.dense = tf.keras.layers.Dense(
            units=5,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, inputs, training=None, mask=None):
        return self.dense(inputs)


'''
net = SimpleNet()
# net.save_weights('easy_checkpoint')
checkpoint = tf.train.Checkpoint(model=net)
checkpoint.save('./path')
'''










