from cgi import print_directory
import os
from pickletools import optimize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

print(tf.__version__)
'''
#定义一个随机数（标量）
random_float = tf.random.uniform(shape=())
print(random_float)

zero_vector = tf.zeros(shape=(2))
print(zero_vector)

zero_array = tf.zeros(shape=(2,2))
print(zero_array)

numpy_array = zero_array.numpy() #将tf中的张量转换成numpy中的数据
print(numpy_array)

'''

# y = ax + b拟合,线性回归 按照下面的x y数据，我们已经知道a=1，b=-1，我们在知道真实值的情况下用tensorflow2.0来进行梯度下降，看看是否正确运行
'''
x = tf.constant([
    [1.],
    [2.],
    [3.],
    [4.],
    [5.]])
y = tf.constant([
    [0.],
    [1.],
    [2.],
    [3.],
    [4.]])

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
vars = [a, b]
num_epochs = 1000
learning_rate = 0.01
optimizer = tf.optimizers.SGD(learning_rate)  # 创建一个梯度下降优化器

for epoch in range(num_epochs):
    with tf.GradientTape() as tape:           # 创建求导记录器(tape: 磁带)
        y_pred = a * x + b
        loss = tf.reduce_sum(tf.square(y_pred - y) / 2)
    grads = tape.gradient(loss, vars)         # 计算loss关于vars的导数（梯度）
    optimizer.apply_gradients(grads_and_vars=zip(grads, vars))  # 将计算出来的导数（梯度）应用优化器的梯度下降

print(vars)
'''

'''
# 通过继承tf.keras.Model来实现简单的线性拟合：y_pred = a * X + b
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


# 计算
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

for i in range(100):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = tf.reduce_mean(tf.square(y_hat - y))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print("a:" + str(model.variables[0].numpy()))
print("b:" + str(model.variables[1].numpy()))

'''


# 多层感知机

class MNISTLoader:
    def __init__(self):
        # mnist = tf.keras.datasets.mnist 这个是从网上下载数据集，我们直接用离线的数据集
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data(
            'C:\\Users\\gongyy\\Desktop\\DNN\\deep_learning_project\\mnist.npz')
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)  # 随机生成一个batch_size大小的列表
        return self.train_data[index, :], self.train_label[index]


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output


# 开始训练
num_epochs = 5
batch_size = 50
learning_rate = 0.001

model = MLP()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches = int(data_loader.num_train_data // batch_size * num_epochs)  # //：整除(向小取整)
list_batch_index = []
list_loss = []
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))

    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
