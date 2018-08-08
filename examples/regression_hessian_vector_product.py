import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

Xc1 = 1.0*np.hstack((np.random.randn(200, 1)+1, np.random.randn(200, 1) - 1))
Xc2 = np.random.multivariate_normal([1, 1], np.array([[0.1, 1.0],
                                                      [1.0, 0.1]]), 200)
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py

X = np.vstack((Xc1, Xc2))
y = np.concatenate((np.zeros(200), np.ones(200)))
idx = np.random.permutation(400)
X = X[idx, :]
y = y[idx].reshape((-1, 1))

# classifier definition
class AdamClassifier:

    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, 2])
        self.labels = tf.placeholder(tf.float32, [None, 1])

        self.W = tf.Variable(tf.random_normal([1, 2]))
        self.b = tf.Variable(tf.random_normal([1]))
        self.h = tf.transpose(tf.sigmoid(tf.matmul(self.W, self.inputs, transpose_b=True)) + self.b)
        self.predictions = tf.reshape(tf.argmax(self.h, 1), shape=[-1, 1])

        self.loss = tf.losses.log_loss(self.labels, self.h)
        #self.loss = tf.reduce_mean(-self.labels*tf.log(self.h + 1e-10) - (1.0 - self.labels)*tf.log(1.0 - self.h + 1e-10))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

class GradOptimizer:

    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, 2])
        self.labels = tf.placeholder(tf.float32, [None, 1])

        self.W = tf.Variable(tf.random_normal([1, 2]))
        self.h = tf.transpose(tf.sigmoid(tf.matmul(self.W, self.inputs, transpose_b=True)))
        self.predictions = tf.reshape(tf.argmax(self.h, 1), shape=[-1, 1])


        # self.loss = tf.reduce_mean(-self.labels*tf.log(self.h + 1e-3) - (1.0 - self.labels)*tf.log(1.0 - self.h + 1e-3))
        # tf.losses.sigmoid_cross_entropy()
        # self.loss = tf.reduce_mean(-self.labels*tf.log(self.h) - (1.0 - self.labels)*tf.log(1 - self.h))
        self.loss = tf.losses.log_loss(self.labels, self.predictions)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)
        self.gradient_holder = tf.placeholder(tf.float32, name='gradient_holder')

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.apply_gradient = optimizer.apply_gradients(zip(self.gradient_holders, tvars))



def run_grad_classifer():
    sess = tf.Session()
    classifier = GradOptimizer()

    sess.run(tf.global_variables_initializer())

    max_steps = 8000

    for step in range(1, max_steps + 1):

        p = sess.run(classifier.predictions, feed_dict={classifier.inputs: X, classifier.labels: y})

        # compute the gradient
        gradients = sess.run(classifier.gradients, feed_dict={classifier.inputs: X, classifier.labels: y})
        feed_dict = dictionary = dict(zip(classifier.gradient_holders, gradients))
        _ = sess.run(classifier.apply_gradient, feed_dict=feed_dict)

        if step % 100 == 0:
            loss = sess.run(classifier.loss, feed_dict={classifier.inputs: X, classifier.labels: y})
            print('step: ' + str(step) + ' loss: ' + str(loss))

    w = classifier.W.eval(sess)[0]

    cline = np.zeros((2, 2))
    cline[0, :] = -4.0 * w
    cline[1, :] = 4.0 * w

    plt.figure()
    plt.plot(Xc1[:, 0], Xc1[:, 1], 'o')
    plt.plot(Xc2[:, 0], Xc2[:, 1], 'o')
    plt.plot(cline[:, 0], cline[:, 1], 'k-', lw=3)
    plt.show()


def run_adam_classifer():

    sess = tf.Session()
    classifier = AdamClassifier()

    sess.run(tf.global_variables_initializer())

    max_steps = 20000

    for step in range(1, max_steps + 1):
        sess.run(classifier.optimizer, feed_dict={classifier.inputs: X, classifier.labels: y})
        if step % 100 == 0:
            w = classifier.W.eval(sess)[0]
            loss = sess.run(classifier.loss, feed_dict={classifier.inputs: X, classifier.labels: y})
            print('step: ' + str(step) + ' loss: ' + str(loss) + ' w ' + str(w))


    w = classifier.W.eval(sess)[0]

    print('w: {}'.format(w))

    cline = np.zeros((2, 2))
    cline[0, :] = -4.0*w
    cline[1, :] = 4.0*w


    plt.figure()
    plt.plot(Xc1[:, 0], Xc1[:, 1], 'o')
    plt.plot(Xc2[:, 0], Xc2[:, 1], 'o')
    plt.plot(cline[:, 0], cline[:, 1], 'k-', lw=3)
    plt.show()


# run_grad_classifer()
run_adam_classifer()