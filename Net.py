#coding=utf-8
import tensorflow as tf
from load_mnist import mnist

mnist = mnist()
mnist.load_mnist('../../data', 'train')

iter_num = 200000      #迭代次数
learning_rate = 0.001  #学习率
batch_size = 64        #每次训练采用的样本数
display = 20           #输出的轮数

input_num = 28*28      #输入图像的像素大小
class_num = 10         #类别的个数
drop_out = 0.8         #剪枝的比率

X = tf.placeholder(tf.float32, shape=(None, input_num))
y = tf.placeholder(tf.float32, shape=[None, class_num])
keep_prob = tf.placeholder(tf.float32)

def conv(input, w, b, name):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='SAME'), b), name=name)

def maxpooling(input, k, name):
    return tf.nn.max_pool(input, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME', name=name)

def norm(input, size, name):
    return tf.nn.lrn(input, size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def AlexNet(X, weights, biases, drop_out):
    X = tf.reshape(X, shape=[-1, 28, 28, 1])

    conv1 = conv(X, weights['conv1'], biases['conv1'], name='conv1')
    pooling1 = maxpooling(conv1, k=2, name='pooling1')
    norm1 = norm(pooling1, size=4, name='norm1')
    norm1 = tf.nn.dropout(norm1, drop_out)

    conv2 = conv(norm1, weights['conv2'], biases['conv2'], name='conv2')
    pooling2 = maxpooling(conv2, k=2, name='pooling2')
    norm2 = norm(pooling2, size=4, name='norm2')
    norm2 = tf.nn.dropout(norm2, drop_out)

    conv3 = conv(norm2, weights['conv3'], biases['conv3'], name='conv3')
    pooling3 = maxpooling(conv3, k=2, name='pooling3')
    norm3 = norm(pooling3, size=4, name='norm3')
    norm3 = tf.nn.dropout(norm3, drop_out)

    fc1 = tf.reshape(norm3, [-1, int(weights['fc1'].get_shape()[0])])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['fc1']) + biases['fc1'], name='fc1')
    fc2 = tf.nn.relu(tf.matmul(fc1, weights['fc2']) + biases['fc2'], name='fc2')
    out = tf.matmul(fc2, weights['out']) + biases['out']
    return out

weights = {
    'conv1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'conv2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'conv3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'fc1': tf.Variable(tf.random_normal([4*4*256, 1024])),
    'fc2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}
biases = {
    'conv1': tf.Variable(tf.random_normal([64])),
    'conv2': tf.Variable(tf.random_normal([128])),
    'conv3': tf.Variable(tf.random_normal([256])),
    'fc1': tf.Variable(tf.random_normal([1024])),
    'fc2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([class_num]))
}

pred = AlexNet(X, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

l2_reg_loss = (tf.nn.l2_loss(weights['conv1']) + tf.nn.l2_loss(weights['conv2']) +
              tf.nn.l2_loss(weights['conv3']) + tf.nn.l2_loss(weights['fc1']) +
              tf.nn.l2_loss(weights['fc2']) + tf.nn.l2_loss(weights['out']) +
              tf.nn.l2_loss(biases['conv1']) + tf.nn.l2_loss(biases['conv2']) +
              tf.nn.l2_loss(biases['conv3']) + tf.nn.l2_loss(biases['fc1']) +
              tf.nn.l2_loss(biases['fc2']) + tf.nn.l2_loss(biases['out']))

loss = cost + l2_reg_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    count = 1
    # Keep training until reach max iterations
    while count*batch_size < iter_num:
        x_batch, y_batch = mnist.random_train_data(batch_size, class_num)
        sess.run(optimizer, feed_dict={X: x_batch, y: y_batch, keep_prob: drop_out})
        if count % display == 0:
            acc = sess.run(accuracy, feed_dict={X: x_batch, y: y_batch, keep_prob: 1.0})
            losses = sess.run(loss, feed_dict={X: x_batch, y: y_batch, keep_prob: 1.0})
            print "iter_num " + str(count*batch_size) + ", Loss= " + "{:.6f}".format(losses)\
                  + ", Training Accuracy= " + "{:.5f}".format(acc)
        count += 1
    print "Finished!"
    mnist.load_mnist('../../data', 'test')
    x_batch, y_batch = mnist.random_train_data(10000, class_num)
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={X: x_batch, y: y_batch, keep_prob: 1.0})


