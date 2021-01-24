import matplotlib.pyplot as plt
import numpy as np


def init_weights(alpha, s, d, same_sign=True, seed=None):
	"""
	alpha = |u_0| * |v_0|
	s = (|v_0| - |u_0|) / (|v_0| + |u_0|)
	"""
	if seed:
		np.random.seed(seed)

	norms_u = []
	norms_v = []
	for alpha_i, s_i in zip(alpha, s):
		norm_u = np.sqrt(alpha_i * (1 - s_i) / (1 + s_i))
		norm_v = np.sqrt(alpha_i * (1 + s_i) / (1 - s_i))
		norms_u.append(norm_u)
		norms_v.append(norm_v)

	u_0 = np.random.normal(size=(d,), loc=0, scale=1)
	u_0 = np.multiply(np.array(norms_u), np.sign(u_0))

	v_0 = np.random.normal(size=(d,), loc=0, scale=1)
	if same_sign:
		v_0 = np.multiply(np.array(norms_v), np.sign(u_0))
	else:
		v_0 = np.multiply(np.array(norms_v), np.sign(v_0))

	return u_0, v_0


def update(u, v, x, y, step_size):
	n, d = x.shape

	y_pred = np.dot(np.multiply(u, v).transpose(), x.transpose())
	margins = np.multiply(y, y_pred)
	gamma = np.min(margins)

	temp = np.exp(gamma - margins)
	grad_r = np.multiply(temp, y) / np.sum(temp)

	u_grad = np.multiply(np.matmul(x.transpose(), grad_r), v)
	v_grad = np.multiply(np.matmul(x.transpose(), grad_r), u)

	u = u + step_size * u_grad
	v = v + step_size * v_grad
	gamma_tilde = gamma - np.log(np.sum(temp) / n)
	return u, v, gamma_tilde, gamma


def minimal_margin(u, v, x, y):
	y_pred = np.dot(np.multiply(u, v).transpose(), x.transpose())
	margins = np.multiply(y, y_pred)
	gamma = np.min(margins)
	return gamma


def normalized_margins(u, v, x, y):
	y_pred = np.dot(np.multiply(u, v).transpose(), x.transpose())
	margins = np.multiply(y, y_pred)
	gamma = np.min(margins)
	return (margins / gamma).reshape(-1,)


def current_training_loss(u, v, x, y):
	n, d = x.shape

	y_pred = np.dot(np.multiply(u, v).transpose(), x.transpose())
	margins = np.multiply(y, y_pred)
	gamma = np.min(margins)

	temp = np.exp(gamma - margins)
	gamma_tilde = gamma - np.log(np.sum(temp) / n)
	return gamma_tilde


def plot_classifier(u, v, x, y):
	xmin = -1
	xmax = 1

	mesh_step = 0.02
	_x1 = np.arange(xmin, xmax, mesh_step)
	_x2 = np.arange(xmin, xmax, mesh_step)
	xx_1, xx_2 = np.meshgrid(_x1, _x2)

	input_ = np.c_[np.ones(xx_1.ravel().shape), xx_1.ravel(), xx_2.ravel()]
	y_pred = np.dot(np.multiply(u, v).transpose(), input_.transpose())

	plt.figure()
	z = np.reshape(np.sign(y_pred), xx_1.shape)
	plt.pcolormesh(xx_1, xx_2, z, cmap='coolwarm')

	plt.scatter([x_[1] for x_, y_ in zip(x, y) if y_ > 0], [x_[2] for x_, y_ in zip(x, y) if y_ > 0], s=100, c='w', marker='+', linewidth=2)
	plt.scatter([x_[1] for x_, y_ in zip(x, y) if y_ < 0], [x_[2] for x_, y_ in zip(x, y) if y_ < 0], s=100, c='w', marker='_', linewidth=2)
	plt.show()
