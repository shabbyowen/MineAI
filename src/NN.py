import tensorflow as tf
import os

n_input = 300
n_hidden_1 = 256
n_hidden_2 = 128
n_output = 1
learning_rate = 0.001

class NN(object):

    def __init__(self):
        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_output])

        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='h1'),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2'),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]), name='out_weight')
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
            'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
            'out': tf.Variable(tf.random_normal([n_output]), name='out_bias')
        }

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()

        if os.path.exists('./model.tf'):
            self.saver.restore(self.sess, "./model.tf")
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)

        # First Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        # Second Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2)

        # Last Output layer with linear activation
        self.pred = tf.nn.softmax(tf.matmul(layer_2, self.weights['out']) + self.biases['out'])
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def predict(self, input_x):
        pred = self.sess.run(self.pred, feed_dict={self.x: input_x})
        return pred[0, 0]


    def train(self, batch_x, batch_y):
        _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})

    def save(self):
        self.saver.save(self.sess, "./model.tf")

if __name__ == '__main__':
    pass
