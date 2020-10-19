import matplotlib.pyplot as plt
import numpy as np


def symmetric_init(alpha, s, m, d, seed=None):
	"""
	alpha = |a_0| + ||w_0||
	s = (|a_0| - ||w_0||) / (|a_0| + ||w_0||)
	"""
	if seed:
		np.random.seed(seed)

	norms_w = []
	norms_a = []
	for alpha_i, s_i in zip(alpha, s):
		norm_w = np.sqrt(alpha_i * (1 - s_i) / (1 + s_i))
		norm_a = np.sqrt(alpha_i * (1 + s_i) / (1 - s_i))
		norms_w.append(norm_w)
		norms_a.append(norm_a)

	w_0 = np.random.normal(size=(m, d), loc=0, scale=1)
	w_0_norms = np.linalg.norm(w_0, axis=1, ord=2)
	w_0 = w_0 / w_0_norms[:, np.newaxis] * np.array(norms_w)[:, np.newaxis]

	a_0 = np.random.normal(size=(1, m), loc=0, scale=1)
	a_0 = np.multiply(np.ones_like(a_0) * np.array(norms_a), (1 * (a_0 > 0) - 0.5) * 2)

	return w_0, a_0


def update(w, a, x, y, epoch, step_size):
	n, d = x.shape

	activations = np.maximum(np.dot(w, x.transpose()), 0)
	y_pred = np.dot(a, activations)
	margins = np.multiply(y, y_pred)
	gamma = np.min(margins)

	temp = np.exp(gamma - margins)
	grad_r = np.multiply(temp, y) / np.sum(temp)

	c_i = 1.0 * (activations > 0)
	w_grad = np.multiply(np.dot(c_i, np.multiply(x, grad_r.transpose())), a.transpose())
	a_grad = np.dot(grad_r, activations.transpose())

	a = a + step_size * a_grad / (np.sqrt(epoch + 1))
	w = w + step_size * w_grad / (np.sqrt(epoch + 1))
	gamma_tilde = gamma - np.log(np.sum(temp) / n)
	return w, a, gamma_tilde, gamma


def minimal_margin(w, a, x, y):
	activations = np.maximum(np.dot(w, x.transpose()), 0)
	y_pred = np.dot(a, activations)
	margins = np.multiply(y, y_pred)
	gamma = np.min(margins)
	return gamma


def plot_classifier(w, a, x, y):
	xmin = -1
	xmax = 1

	mesh_step = 0.02
	_x1 = np.arange(xmin, xmax, mesh_step)
	_x2 = np.arange(xmin, xmax, mesh_step)
	xx_1, xx_2 = np.meshgrid(_x1, _x2)

	input_ = np.c_[np.ones(xx_1.ravel().shape), xx_1.ravel(), xx_2.ravel()]
	activations = np.maximum(np.dot(w, input_.transpose()), 0)
	y_pred = np.dot(a, activations)

	plt.figure()
	z = np.reshape(np.sign(y_pred), xx_1.shape)
	plt.pcolormesh(xx_1, xx_2, z)

	plt.scatter([x_[1] for x_, y_ in zip(x, y) if y_ > 0], [x_[2] for x_, y_ in zip(x, y) if y_ > 0])
	plt.scatter([x_[1] for x_, y_ in zip(x, y) if y_ < 0], [x_[2] for x_, y_ in zip(x, y) if y_ < 0])
	plt.show()
