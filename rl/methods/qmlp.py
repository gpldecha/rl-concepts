import os
import numpy as np
import tensorflow as tf

# check https://github.com/tflearn/tflearn/blob/master/examples/reinforcement_learning/atari_1step_qlearning.py

class QMLP:

    def __init__(self, n_input, hidden_layers, n_output):

        num_hidden_layers = len(hidden_layers)

        self.X = tf.placeholder(tf.float32, [None, n_input])
        self.Y = tf.placeholder(tf.float32, [None, n_output])

        if num_hidden_layers == 1:
            self.parameters = {
                'h1': tf.Variable(tf.random_normal([n_input, hidden_layers[0]])),
                'wout': tf.Variable(tf.random_normal([hidden_layers[0], n_output])),
                'b1': tf.Variable(tf.random_normal([hidden_layers[0]])),
                'bout': tf.Variable(tf.random_normal([n_output]))
            }
        else:
            self.parameters = dict()
            self.parameters['h1'] = tf.Variable(tf.random_normal([n_input, hidden_layers[0]]))
            self.parameters['b1'] = tf.Variable(tf.random_normal([hidden_layers[0]]))

            for i in range(1, num_hidden_layers):
                self.parameters['h'+str(i+1)] = tf.Variable(tf.random_normal([hidden_layers[i-1], hidden_layers[i]]))
                self.parameters['b'+str(i+1)] = tf.Variable(tf.random_normal([hidden_layers[i]]))

            self.parameters['wout'] = tf.Variable(tf.random_normal([hidden_layers[-1], n_output]))
            self.parameters['bout'] = tf.Variable(tf.random_normal([n_output]))

        if num_hidden_layers == 1:
            layer = tf.add(tf.matmul(self.X, self.parameters['h1']), self.parameters['b1'])
            layer = tf.sigmoid(layer, name='layer_sigmoid')
        else:
            layer = tf.add(tf.matmul(self.X, self.parameters['h1']), self.parameters['b1'])
            layer = tf.sigmoid(layer, name='layer1_sigmoid')
            for i in range(2, max(num_hidden_layers+1, 2)):
                layer = tf.add(tf.matmul(layer, self.parameters['h'+str(i)]), self.parameters['b'+str(i)])
                layer = tf.sigmoid(layer, name='layer'+str(i)+'_sigmoid')

        self.q_values = tf.matmul(layer, self.parameters['wout']) + self.parameters['bout']
        self.max_action = tf.argmax(self.q_values, 1)

        self.loss = tf.nn.l2_loss(self.q_values - self.Y)

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.saver = None
        self.is_saving = True
        self.save_step_interval = 1000
        self.ckpt_name = 'qmlp'
        self.y = None

        self.logs_path = os.path.dirname(os.path.realpath(__file__)) + '/../models'

        self._is_initialised = False
        self.display_step = 1000

    def initialise(self, sess):
        if not self._is_initialised:
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(self.parameters)
            self._is_initialised = True

    def _save_model(self, sess, training_step):
        if self.is_saving:
            checkpoint_path = os.path.join(self.logs_path, self.ckpt_name + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            self.saver.save(sess, checkpoint_path, global_step=training_step)

    def train(self, sess, s, y, max_steps):
        self.initialise(sess)

        for step in range(1, max_steps+1):
            sess.run(self.optimizer, feed_dict={self.X: s, self.Y: y})

            if step % self.save_step_interval == 0:
                self._save_model(sess, step)

            if step % self.display_step == 0 or step == 1:
                loss = sess.run([self.loss], feed_dict={self.X: s, self.Y: y})
                # print('step: ' + str(step) + ' loss: ' + str(loss))

        self._save_model(sess, max_steps)

    def predict(self, sess, input):
        self.initialise(sess)
        output = sess.run([self.q_values], feed_dict={self.X: input})
        return output[0]

    def max_a(self, sess, state):
        self.initialise(sess)
        action, q_values = sess.run([self.max_action, self.q_values], feed_dict={self.X: state})
        return action[0], q_values[0]

    def update_parameters(self, sess, qmlp, tau):
        for idx, var in qmlp.parameters.iteritems():
            sess.run(self.parameters[idx].assign((var.value()*tau) + ((1.0-tau)*self.parameters[idx].value())))

    def get_parameter(self, sess, name):
        value = sess.run(self.parameters[name])
        if value.ndim == 1:
            return value.reshape((len(value), 1))
        else:
            return value

    def get_targets(self, sess, discount, s, r, sp, done):
        if (self.y is None) or (len(self.y) != len(r)):
            self.y = np.empty(shape=(len(r), 3))

        idx_done = done == 1
        idx = done == 0

        self.y[idx_done] = r[idx_done]
        actions, values = self.max_a(sess=sess, state=sp[idx, :])

        # self.y[idx] = r[idx] + discount*values[actions]
        self.y[idx, :] = r[idx] + discount*values

        return self.y, s