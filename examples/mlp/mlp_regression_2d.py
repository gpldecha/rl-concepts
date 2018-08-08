import tensorflow as tf
from rl.methods.qmlp import QMLP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *
from matplotlib import cm

# Generate training Data

num_samples = 100
X = np.random.uniform(0, 1, (2, num_samples))
y = np.sin(5.0 * X[0, :]) * np.cos(5.0 * X[1, :])/5.0 + np.random.normal(0, 0.025, num_samples)
X = X.transpose()
y = y.reshape((num_samples, 1))

print('X: ', X.shape[0], 'x', X.shape[1])
print('y: ', y.shape[0], 'x', y.shape[1])

mlp = QMLP(n_input=2, hidden_layers=[20, 20], n_output=1)
sess = tf.Session()

# Train

mlp.train(sess=sess, s=X, y=y, max_steps=10000)


# Test

x_surf = np.arange(0, 1, 0.01)
y_surf = np.arange(0, 1, 0.01)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = np.sin(5.0 * x_surf) * np.cos(5.0 * y_surf)/5.0

Xq = np.vstack((x_surf.flatten(), y_surf.flatten())).transpose()

print('Xq: ', Xq.shape[0], 'x', Xq.shape[1])
yq = mlp.predict(sess=sess, input=Xq)
print('yq: ', yq.shape[0], 'x', yq.shape[1])


# Plot true function with training data

plt.close('all')
fig = plt.figure()
ax = fig.gca(projection='3d')               # to work in 3d
plt.hold(True)

ax.plot_surface(x_surf, y_surf, yq.reshape(z_surf.shape), alpha=0.8)    # plot a 3d surface plot

ax.scatter(X[:, 0], X[:, 1], y, s=20)

ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_zlabel('z label')

plt.show()
