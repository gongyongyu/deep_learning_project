# coding:utf-8
import math

import numpy as np
import tensorflow as tf
import argparse
import cv2
import matplotlib.pyplot as plt

'''
net = SimpleNet()
# net.save_weights('easy_checkpoint')
checkpoint = tf.train.Checkpoint(model=net)
checkpoint.save('./path')
'''

'''
init_array = np.array([1., 0., 0.])
transfer_matrix = np.array([[0.9, 0.075, 0.025],
                            [0.15, 0.8, 0.05],
                            [0.25, 0.25, 0.5]
                            ])
sum_1 = np.sum(transfer_matrix, axis=0, keepdims=True)
sum_2 = np.sum(transfer_matrix, axis=1, keepdims=True)
print(sum_1)
print(sum_2)

res = init_array
for i in range(25):
    print("i= %d" % i)
    print(res)
    res = np.dot(res, transfer_matrix)
'''

if __name__ == '__main__':
    # tensorflow中的tf.matmul()与tf.multiply()的区别
    '''
    matrix_a = tf.constant([
        [1, 2],
        [3, 4]])
    matrix_b = tf.constant([
        [5, 6],
        [7, 8]])
    matrix_c = tf.matmul(matrix_a, matrix_b)
    print(matrix_c.numpy())
    print(np.dot(matrix_a.numpy(), matrix_b.numpy()))
    '''
    # 自动求导机制
    '''
    X = tf.constant([
        [1., 2.],
        [3., 4.]
    ])

    y = tf.constant([
        [1.],
        [2.]
    ])

    w = tf.Variable([
        [1.],
        [2.]
    ])

    b = tf.Variable(initial_value=1.)

    with tf.GradientTape() as tape:
        loss = tf.square(tf.matmul(X, w) + b - y)
    loss_grad = tape.gradient(loss, [w, b])
    print(loss, loss_grad)
    '''

'''
    # y = ax + b 拟合x及y
    import numpy as np
    import matplotlib.pyplot as plt

    x_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
    y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

    # 归一化操作
    x = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

    # 使用梯度下降来拟合
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    a = tf.Variable(initial_value=0.)
    b = tf.Variable(initial_value=0.)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)  # 实例化SGD优化器
    num_epochs = 1000

    x_ = []
    y_ = []
    for i in range(num_epochs):
        with tf.GradientTape() as tape:
            y_hat = a * x + b
            loss = tf.reduce_sum(tf.square(y_hat - y))
        grads = tape.gradient(loss, [a, b])
        optimizer.apply_gradients(grads_and_vars=zip(grads, [a, b]))
        x_.append(i)
        y_.append(loss)

        # plt循环体内持续画图
        plt.ion()   # 打开交互模式
        plt.plot(x_, y_)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()
        plt.pause(0.02)
        plt.clf()   # 清除图像

        if i % 10 == 0:
            print(loss.numpy(), [a.numpy(), b.numpy()])
'''


# 模型（Model）与层（Layer）
class Linear(tf.keras.Model):
    def __init__(self):
        super(Linear, self).__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


def train():
    x_raw = np.array([[2013], [2014], [2015], [2016], [2017]], dtype=np.float32)
    y_raw = np.array([[12000], [14000], [15000], [16500], [17500]], dtype=np.float32)

    # 归一化操作
    x = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

    # 使用梯度下降来拟合
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    model = Linear()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    num_epochs = 1000
    checkpoint = tf.train.Checkpoint(myLinearModel=model)  # 实例化Checkpoint，设置保存对象为model，键为myLinearModel
    x_ = []
    y_ = []

    for i in range(num_epochs):
        with tf.GradientTape() as tape:
            y_hat = model(x)
            loss = tf.reduce_sum(tf.square(y_hat - y))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        x_.append(i)
        y_.append(loss)
        plt.ion()
        plt.plot(x_, y_)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
        plt.pause(0.01)
        if i == 4999:
            plt.savefig('fig')
        else:
            plt.clf()

        if i % 100 == 0:  # 每隔100个Batch保存一次
            print(loss.numpy(), [model.variables[0].numpy(), model.variables[1].numpy()])
            path = checkpoint.save('./save/model.ckpt')  # 保存模型参数到文件,save()方法返回路径
            print("model saved to %s" % path)


def inference(x_raw):
    x = (x_raw - 2013.) / (2017. - 2013.)

    model_to_be_restored = Linear()
    # 实例化Checkpoint，设置恢复对象为新建立的模型model_to_be_restored
    checkpoint = tf.train.Checkpoint(myLinearModel=model_to_be_restored)
    checkpoint.restore(tf.train.latest_checkpoint('./save'))  # 从文件恢复模型参数
    y_hat = model_to_be_restored.predict(x)

    y_hat_raw = y_hat * 5500. + 12000.
    return y_hat_raw


if __name__ == '__main__':
    # train()
    print(inference(tf.constant([[2022.]])))
