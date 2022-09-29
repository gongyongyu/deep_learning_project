# coding:utf-8
import numpy as np
import tensorflow as tf
import argparse
from tensorflow2_test import MLP
from tensorflow2_test import MNISTLoader

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument('--num_epochs', default=5)
parser.add_argument('--batch_size', default=50)
parser.add_argument('--learning_rate', default=0.001)
args = parser.parse_args()
data_loader = MNISTLoader()


def train():
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    # 一个batch执行num_epochs次梯度下降，总共的梯度下降次数就能按照下面的计算出来
    num_batches = int(data_loader.num_train_data // args.batch_size * args.num_epochs)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)  # 实例化Checkpoint，设置保存对象为model
    for batch_index in range(1, num_batches + 1):
        x, y = data_loader.get_batch(args.batch_size)
        with tf.GradientTape() as tape:
            y_hat = model(x)  # 描述y_hat（输出）与输入（x）的关系
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_hat)  # 描述loss的计算方式
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if batch_index % 100 == 0:  # 每隔100个Batch保存一次
            path = checkpoint.save('./save/model.ckpt')  # 保存模型参数到文件
            print("model saved to %s" % path)


def test():
    model_to_be_restored = MLP()
    # 实例化Checkpoint，设置恢复对象为新建立的模型model_to_be_restored
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)
    checkpoint.restore(tf.train.latest_checkpoint('./save'))  # 从文件恢复模型参数
    y_hat = np.argmax(model_to_be_restored.predict(data_loader.test_data), axis=-1)
    print("test accuracy: %f" % (sum(y_hat == data_loader.test_label) / data_loader.num_test_data))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    if args.mode == 'test':
        test()
