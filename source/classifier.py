import matplotlib.pyplot as plt
import numpy as np


def symmetric_init(alpha, s, d, symmetric=True, seed=None):
	if seed:
		np.random.seed(seed)

	norm_w = np.sqrt(alpha * (1 - s) / (1 + s))
	norm_a = np.sqrt(alpha * (1 + s) / (1 - s))

	u_0 = np.random.normal(size=(d, 1), loc=0, scale=1)
	u_0_norm = np.linalg.norm(u_0, ord=2)
	u_0 = u_0 / u_0_norm * norm_w

	v_0 = np.random.normal(size=(d, 1), loc=0, scale=1)
	v_0_norm = np.linalg.norm(v_0, ord=2)
	v_0 = v_0 / v_0_norm * norm_w

	if symmetric:
		v_0 = u_0

	a_0 = (1 * (np.random.normal(size=(1,), loc=0, scale=1) > 0) - 0.5) * 2 * norm_a

	return u_0, v_0, a_0


def update(u, v, a, x, y, step_size):
	d, n = x.shape

	y_pred = np.matmul((u - v).transpose(), x) * a
	grad_r = -(y_pred - y)/n

	grad_xr = np.matmul(x, grad_r.transpose())

	u_grad = grad_xr * a
	v_grad = -grad_xr * a
	a_grad = np.matmul((u - v).transpose(), grad_xr)

	u = u + step_size * u_grad
	v = v + step_size * v_grad
	a = a + step_size * a_grad

	return u, v, a


def update_with_relu(u, v, a, x, y, step_size):
	d, n = x.shape

	activations = np.maximum(np.matmul((u - v).transpose(), x), 0)
	c_n = 1.0 * (activations > 0)
	y_pred = activations * a
	grad_r = -(y_pred - y)/n
	grad_r = np.multiply(grad_r, c_n)

	grad_xr = np.matmul(x, grad_r.transpose())

	u_grad = grad_xr * a
	v_grad = -grad_xr * a
	a_grad = np.matmul((u - v).transpose(), grad_xr)

	u = u + step_size * u_grad
	v = v + step_size * v_grad
	a = a + step_size * a_grad

	return u, v, a


def current_training_loss(u, v, a, x, y):
	y_pred = np.matmul((u - v).transpose(), x) * a
	return 0.5 * np.linalg.norm(y - y_pred, ord=2) ** 2


def current_training_loss_with_relu(u, v, a, x, y):
	activations = np.maximum(np.matmul((u - v).transpose(), x), 0)
	y_pred = activations * a
	return 0.5 * np.linalg.norm(y - y_pred, ord=2) ** 2