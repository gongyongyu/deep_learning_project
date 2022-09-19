from cgi import print_directory
import os
from pickletools import optimize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np

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

