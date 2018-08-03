import tensorflow as tf


class MLP:

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
                self.parameters['h' + str(i + 1)] = tf.Variable(
                    tf.random_normal([hidden_layers[i - 1], hidden_layers[i]]))
                self.parameters['b' + str(i + 1)] = tf.Variable(tf.random_normal([hidden_layers[i]]))

            self.parameters['wout'] = tf.Variable(tf.random_normal([hidden_layers[-1], n_output]))
            self.parameters['bout'] = tf.Variable(tf.random_normal([n_output]))

        if num_hidden_layers == 1:
            layer = tf.add(tf.matmul(self.X, self.parameters['h1']), self.parameters['b1'])
            layer = tf.sigmoid(layer, name='layer_sigmoid')
        else:
            layer = tf.add(tf.matmul(self.X, self.parameters['h1']), self.parameters['b1'])
            layer = tf.sigmoid(layer, name='layer1_sigmoid')
            for i in range(2, max(num_hidden_layers + 1, 2)):
                layer = tf.add(tf.matmul(layer, self.parameters['h' + str(i)]), self.parameters['b' + str(i)])
                layer = tf.sigmoid(layer, name='layer' + str(i) + '_sigmoid')

        self.logits = tf.matmul(layer, self.parameters['wout']) + self.parameters['bout']
        # Softmax
        self.action_probabilities = tf.nn.softmax(self.logits, name='action_probs')