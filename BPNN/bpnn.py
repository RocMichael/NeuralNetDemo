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


class BPNeuralNetwork:
    def __init__(self):
        self.session = tf.Session()
        self.x_src = None
        self.y_src = None
        self.loss = None
        self.trainer = None
        self.layers = []

    def __del__(self):
        self.session.close()

    def train(self, x_data, y_data, limit=10000, learn_rate=0.1):
        # build network
        self.x_src = tf.placeholder(tf.float32, [None, 1])
        self.y_src = tf.placeholder(tf.float32, [None, 1])
        self.layers.append(add_layer(self.x_src, 1, 10, activity_function=tf.nn.relu))
        self.layers.append(add_layer(self.layers[0], 10, 1, activity_function=None))
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square((self.y_src - self.layers[1])), reduction_indices=[1]))
        self.trainer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
        initer = tf.initialize_all_variables()
        # do training
        self.session.run(initer)
        for i in range(limit):
            self.session.run(self.trainer, feed_dict={self.x_src: x_data, self.y_src: y_data})

    def predict(self, x_data):
        return self.session.run(self.layers[-1], feed_dict={self.x_src: x_data})

    def test(self):
        x_data = np.array([[1, 0, 0, 1, 0, 1, 0, 1, 0, 1]]).transpose()
        y_data = np.array([[0, 1, 1, 0, 1, 0, 1, 0, 1, 0]]).transpose()
        test_data = np.array([[0, 1]]).transpose()
        self.train(x_data, y_data)
        print self.predict(test_data)

nn = BPNeuralNetwork()
nn.test()