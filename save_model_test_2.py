# encoding : utf-8
import tensorflow as tf
from tensorflow2_test import MNISTLoader

num_epochs = 5
batch_size = 50
learning_rate = 0.001

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=100,
                          use_bias=True,
                          kernel_initializer=tf.zeros_initializer(),
                          bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(units=10,
                          use_bias=True,
                          kernel_initializer=tf.zeros_initializer(),
                          bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Softmax()
])

data_loader = MNISTLoader()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_crossentropy]
)

model.fit(data_loader.train_data, data_loader.train_label, batch_size=batch_size, epochs=num_epochs)
tf.saved_model.save(model, ".\\save\\saveModel\\1")
