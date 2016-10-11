import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activity_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    basis = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    weights_plus_b = tf.matmul(inputs, weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans


# x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) + 0.5 + noise

# x_data = [
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ]
#

x_data = np.array([[1, 0, 0, 1, 0, 1, 0, 1, 0, 1]]).transpose()

y_data = np.array([[0, 1, 1, 0, 1, 0, 1, 0, 1, 0]]).transpose()

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activity_function=tf.nn.relu)
l2 = add_layer(l1, 10, 1, activity_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - l2)), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()


with tf.Session() as session:
    session.run(init)
    for i in range(10000):
        session.run(train, feed_dict={xs: x_data, ys: y_data})
        # if i % 50 == 0:
        #     print session.run(loss, feed_dict={xs: x_data, ys: y_data})
    print session.run(l2, feed_dict={xs: [1, 0]})


