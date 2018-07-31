import tensorflow as tf


class MLP:

    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_output):

        self.X = tf.placeholder("float", [None, n_input])
        self.Y = tf.placeholder("float", [None, n_output])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_output]))
        }

        layer_1 = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.sigmoid(layer_1, name='layer1_sigmoid')

        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.sigmoid(layer_2, name='layer2_sigmoid')

        self.prediction = tf.matmul(layer_2, self.weights['out']) + self.biases['out']

        self.loss = tf.nn.l2_loss(self.prediction - self.Y)

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self._is_initialised = False
        self.display_step = 50

    def train(self, sess, input, output, max_steps):
        if not self._is_initialised:
            sess.run(tf.global_variables_initializer())

        for step in range(1, max_steps+1):
            sess.run(self.optimizer, feed_dict={self.X: input, self.Y: output})

            if step % self.display_step == 0 or step == 1:
                loss = sess.run([self.loss], feed_dict={self.X: input, self.Y: output})
                print('step: ' + str(step) + ' loss: ' + str(loss))

    def predict(self, sess, input):
        output = sess.run([self.prediction], feed_dict={self.X: input})
        return output[0]



if __name__ == "__main__":
    print('testing mlp')

    mpl = MLP(n_input=2, n_hidden_1=40, n_hidden_2=40, n_output=1)