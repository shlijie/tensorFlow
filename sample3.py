import tensorflow as tf

# 定义添加神经层
def add_layer(inputs, in_size, out_size, activation_function=None):

    # 定义weights和biases
    # weights为一个in_size行, out_size列的随机变量矩阵
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))

    # biases的推荐值不为0, 在0向量的基础上又加了0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    # 定义神经网络未激活的值
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # 激励函数为None时，输出就是当前的预测值
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    
    # 返回输出
    return outputs
