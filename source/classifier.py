import matplotlib.pyplot as plt
import numpy as np


def symmetric_init(alpha, s, d, seed=None):
	if seed:
		np.random.seed(seed)

	norm_u = np.sqrt(alpha * (1 - s) / (1 + s))
	norm_v = np.sqrt(alpha * (1 + s) / (1 - s))

	u_0 = np.ones((d, 1)) * norm_u
	v_0 = np.ones((d, 1)) * norm_v

	u_p_0 = u_n_0 = u_0
	v_p_0 = v_n_0 = v_0
	return u_p_0, v_p_0, u_n_0, v_n_0


def update(u_p, v_p, u_n, v_n, x, y, step_size):
	d, n = x.shape

	w = np.multiply(u_p, v_p) - np.multiply(u_n, v_n)
	y_pred = np.matmul(w.transpose(), x)
	grad_r = -(y_pred - y)/n

	grad_xr = np.matmul(x, grad_r.transpose())

	u_p_grad = np.multiply(grad_xr, v_p)
	v_p_grad = np.multiply(grad_xr, u_p)
	u_n_grad = -np.multiply(grad_xr, v_n)
	v_n_grad = -np.multiply(grad_xr, u_n)

	u_p = u_p + step_size * u_p_grad
	v_p = v_p + step_size * v_p_grad
	u_n = u_n + step_size * u_n_grad
	v_n = v_n + step_size * v_n_grad

	return u_p, v_p, u_n, v_n


def current_training_loss(u_p, v_p, u_n, v_n, x, y):
	d, n = x.shape
	w = np.multiply(u_p, v_p) - np.multiply(u_n, v_n)
	y_pred = np.matmul(w.transpose(), x)
	return np.linalg.norm(y - y_pred, ord=2) ** 2 / n
