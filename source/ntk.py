import matplotlib.pyplot as plt
import numpy as np


def cosine_sim(u, v):
	return np.dot(u.reshape(-1,), v.reshape(-1,)) / np.linalg.norm(u) / np.linalg.norm(v)


def update_ntk(w, a, w_0, a_0, x, y, epoch, step_size):
	n, d = x.shape

	activations = np.maximum(np.dot(w, x.transpose()), 0)
	activations_0 = np.maximum(np.dot(w_0, x.transpose()), 0)
	c_i_0 = 1.0 * (activations_0 > 0)

	y_pred = -np.dot(a_0, activations_0) + np.dot(a, activations_0) + np.dot(a_0, np.multiply(activations, c_i_0))

	margins = np.multiply(y, y_pred)
	gamma = np.min(margins)

	temp = np.exp(gamma - margins)
	grad_r = np.multiply(temp, y) / np.sum(temp)

	w_grad = np.multiply(np.dot(c_i_0, np.multiply(x, grad_r.transpose())), a_0.transpose())
	a_grad = np.dot(grad_r, activations_0.transpose())

	a = a + step_size * a_grad / (np.sqrt(epoch + 1))
	w = w + step_size * w_grad / (np.sqrt(epoch + 1))
	gamma_tilde = gamma - np.log(np.sum(temp) / n)
	return w, a, gamma_tilde, gamma


def plot_classifier_ntk(w, a, w_0, a_0, x, y):
	xmin = -1
	xmax = 1

	mesh_step = 0.02
	_x1 = np.arange(xmin, xmax, mesh_step)
	_x2 = np.arange(xmin, xmax, mesh_step)
	xx_1, xx_2 = np.meshgrid(_x1, _x2)

	input_ = np.c_[np.ones(xx_1.ravel().shape), xx_1.ravel(), xx_2.ravel()]
	activations = np.maximum(np.dot(w, input_.transpose()), 0)
	activations_0 = np.maximum(np.dot(w_0, input_.transpose()), 0)
	c_i_0 = 1.0 * (activations_0 > 0)
	y_pred = -np.dot(a_0, activations_0) + np.dot(a, activations_0) + np.dot(a_0, np.multiply(activations, c_i_0))

	plt.figure()
	z = np.reshape(np.sign(y_pred), xx_1.shape)
	plt.pcolormesh(xx_1, xx_2, z)

	plt.scatter([x_[1] for x_, y_ in zip(x, y) if y_ > 0], [x_[2] for x_, y_ in zip(x, y) if y_ > 0])
	plt.scatter([x_[1] for x_, y_ in zip(x, y) if y_ < 0], [x_[2] for x_, y_ in zip(x, y) if y_ < 0])
	plt.show()
