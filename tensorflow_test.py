#coding:utf-8

''' J(w)=w^2-10w+25 
我们用tensorflow来优化最小值, 当然我们能够直接算出w=5时J最小
'''

import numpy as np
import tensorflow as tf # version = 1.11.0


'''start-
coefficient = np.array([[1.], [-20.], [100.]])
w = tf.Variable(0, dtype=tf.float32) #将w初始化为0，我们将要优化的参数叫做 Variable
x = tf.placeholder(tf.float32, [3,1]) #定义placeholder，暗示一会会提供数据，一般将训练数据设置成placeholder
#cost = w**2 - 10*w + 25 #定义cost 函数，因为w已经被定义为tf的变量，因此这些计算符号都被tf重载过了，能直接使用
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost) #定义使用梯度下降进行train，并将学习速率设置为0.01

#以下是惯用表达
init = tf.global_variables_initializer()
#session = tf.Session()
#session.run(init) #开始运行梯度下降
#print(session.run(w))
with tf.Session() as session:  #用with语句替代上面三行代码
    session.run(init)
    print(session.run(w))
    for i in range(1000):
        session.run(train, feed_dict={x:coefficient}) #运行一步梯度下降

-end'''



print("--------------")






