import tensorflow as tf
from rl.methods.qmlp import QMLP
import numpy as np
import matplotlib.pyplot as plt

# Generate some test data (1D)
num_samples = 200
X = np.concatenate((np.linspace(-10, -3, num_samples/2) , np.linspace(3, 10, num_samples/2)), axis=0)
y = 10.0*np.sin(0.5 * X) + np.random.normal(0, 1.1, num_samples)
X = X.reshape((num_samples, 1))
y = y.reshape((num_samples, 1))


print X.shape

print('X: ', X.shape[0], 'x', X.shape[1])
print('y: ', y.shape[0], 'x', y.shape[1])


mlp = QMLP(n_input=1, hidden_layers=[5, 5, 5], n_output=1)

sess = tf.Session()

print 'start training'
mlp.train(sess=sess, s=X, y=y, max_steps=10000)
print 'training finished'

Xq = np.linspace(-10, 10, num_samples)
Xq = Xq.reshape((num_samples, 1))

print('Xq: ', Xq.shape[0], 'x', Xq.shape[1])
yq = mlp.predict(sess=sess, input=Xq)
print('yq: ', yq.shape[0], 'x', yq.shape[1])


fig = plt.figure(facecolor='white')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(X, y, 'o')
plt.plot(Xq, yq, '-r', lw=2)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('MLP', fontsize=20)
plt.axis('equal')

plt.show()