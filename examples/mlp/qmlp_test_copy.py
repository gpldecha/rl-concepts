import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from rl.methods.qmlp import QMLP


num_samples = 200
X = np.concatenate((np.linspace(-10, -3, num_samples/2) , np.linspace(3, 10, num_samples/2)), axis=0)
y = 10.0*np.sin(0.5 * X) + np.random.normal(0, 1.1, num_samples)
X = X.reshape((num_samples, 1))
y = y.reshape((num_samples, 1))

sess = tf.Session()


qmlp1 = QMLP(n_input=1, hidden_layers=[5, 5], n_output=1)
qmlp1.is_saving = False
qmlp1.initialise(sess)

h2 = qmlp1.get_parameter(sess, 'h2')



# qmlp2 = QMLP(n_input=1, hidden_layers=[10], n_output=1)
# qmlp2.is_saving = False
# qmlp2.initialise(sess)
#
# print 'qm1 h1: \n', qmlp1.get_parameter(sess, 'h1')
# print 'qm2 h1: \n', qmlp2.get_parameter(sess, 'h1')

print 'start training'
qmlp1.train(sess=sess, s=X, y=y, max_steps=40000)
print 'training finished'
#
# print 'qm1 h1: \n', qmlp1.get_parameter(sess, 'h1')
# print 'qm2 h1: \n', qmlp2.get_parameter(sess, 'h1')
#
# print 'update parameters\n\n'
#
# qmlp2.update_parameters(sess, qmlp1, 1.0)
#
# print 'qm1 h1: \n', qmlp1.get_parameter(sess, 'h1')
# print 'qm2 h1: \n', qmlp2.get_parameter(sess, 'h1')
#
#
plt.figure()
plt.subplot(2, 1, 1)
cax = plt.imshow(h2, interpolation='None', cmap='seismic')
plt.axis('off')
plt.colorbar(cax)
plt.subplot(2, 1, 2)
cax = plt.imshow(qmlp1.get_parameter(sess, 'h2'), interpolation='None', cmap='seismic')
plt.axis('off')
plt.colorbar(cax)
plt.show(False)

Xq = np.linspace(-10, 10, num_samples)
Xq = Xq.reshape((num_samples, 1))

print('Xq: ', Xq.shape[0], 'x', Xq.shape[1])
yq = qmlp1.predict(sess=sess, input=Xq)
print('yq: ', yq.shape[0], 'x', yq.shape[1])


fig = plt.figure(facecolor='white')
plt.plot(X, y, 'o')
plt.plot(Xq, yq, '-r', lw=2)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('MLP', fontsize=20)
plt.axis('equal')
plt.show()