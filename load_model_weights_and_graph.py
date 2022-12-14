# encoding : utf-8

import tensorflow as tf
import os
from tensorflow2_test import MNISTLoader

batch_size = 50
model = tf.saved_model.load('.\\save\\saveModel\\1')
data_loader = MNISTLoader()
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)

for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_hat = model(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_hat)
print("test accuracy: %f" % sparse_categorical_accuracy.result())
