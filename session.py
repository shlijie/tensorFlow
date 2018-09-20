import tensorflow as tf

# create two matrixes
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

# 两个 matrix 矩阵相乘的结果
product = tf.matmul(matrix1,matrix2)

# 使用 Session 来激活 product 并得到计算结果
# 有两种形式使用会话控制 Session
# method 1
sess = tf.Session()
result = sess.run(product)
# print('matrix1')
# print(matrix1)
# print('matrix2')
# print(matrix2)
print('product')
print(result)
sess.close()
# [[12]]

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print('product')
    print(result2)
# [[12]]
