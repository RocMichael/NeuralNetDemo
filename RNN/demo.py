import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

batch_size = 1
n_steps = 3
n_input = 2
n_hidden = 5
n_classes = 2
learning_rate = 0.05
limit = 1000

# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
input_layer = [tf.placeholder("float", [n_steps, n_input])]  # * batch_size
label_layer = tf.placeholder("float", [n_steps, n_classes])

weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))
lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, states = rnn.rnn(lstm_cell, input_layer, dtype=tf.float32)
prediction = tf.matmul(outputs[-1], weights) + biases
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, label_layer))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
initer = tf.initialize_all_variables()

# Current data input shape: (batch_size, n_steps, n_input)
train_x = np.array([[1, 1, 1, 1, 1, 1]])
train_x = train_x.reshape((batch_size, n_steps, n_input))
train_y = np.array([[1, 1, 1, 1, 1, 1]])
train_y = train_y.reshape((n_steps, n_input))
test_x = np.array([[]])

with tf.Session() as session:
    session.run(initer)
    for i in range(limit):
        session.run(optimizer, feed_dict={input_layer[0]: train_x[0], label_layer: train_y})
    print session.run(prediction, feed_dict={input_layer[0]: train_x[0]})