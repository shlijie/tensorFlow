import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
# y=Weights * x + biases, 神经网络也就是学着把 Weights 变成 0.1, biases 变成 0.3
y_data = x_data*0.1 + 0.3

# 搭建模型
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases

# 计算误差
loss = tf.reduce_mean(tf.square(y-y_data))

# 传播误差
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 训练
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)          # Very important

# 用 Session 来 run 每一次 training 的数据. 逐步提升神经网络的预测准确性.
for step in range(201):
    sess.run(train)
    # if step % 20 == 0:
    print(step, sess.run(Weights), sess.run(biases))

