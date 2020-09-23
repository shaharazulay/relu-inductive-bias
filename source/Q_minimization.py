import numpy as np
from scipy.optimize import minimize


def flatten(w, a):
	return np.hstack([w.flatten(), a.flatten()])


def restore(v, m, d):
	w = v[: m * d].reshape(m, d)
	a = v[m * d:].reshape(m,)
	return w, a


def q_func(x):
	return 0.5 * x * (np.log(x) - 1)


def Q_func(w, a, mu):
	"""
	:param w: matrix of shape m x d representing the first layer w
	:param a: vector of shape m x 1 representing the second layer a
	:param mu: vector of shape m x 1 for each ReLU.
	"""
	f = 0
	for w_i, a_i, mu_i in zip(w, a, mu):
		f += mu_i * q_func(np.linalg.norm(a_i * w_i, ord=2) / mu_i)
	return f


def Q_func_objective(v, mu, m, d):
	w, a = restore(v, m, d)
	return Q_func(w, a, mu)


def margin_constraint(v, x, y, m, d):
	"""
	for all n, y_n * sum(a_i[w_i^T x_n]+) >= 1
	"""
	w, a = restore(v, m, d)

	activations = np.maximum(np.dot(w, x.transpose()), 0)
	y_pred = np.dot(a, activations)
	margins = np.multiply(y, y_pred)

	return margins - 1


def calc_w_tilde_norms(w, a):
	w_norms = np.linalg.norm(w, ord=2, axis=1)
	return np.multiply(np.abs(a), w_norms).reshape(-1,)


def solver(x, y, w_0, a_0, m, d, obj='L1', mu=None, optim_tol=1e-6):
	v_0 = flatten(w_0, a_0)

	cons = {'type': 'ineq', 'fun': lambda v: margin_constraint(v, x, y, m, d)}

	if obj == 'L1':
		objective = lambda v: np.linalg.norm(v, ord=2)
	elif obj == 'Q':
		objective = lambda v: Q_func_objective(v, mu, m, d)
	else:
		raise ValueError('objective not supported.')

	sol = minimize(
		fun=objective,
		x0=v_0,
		constraints=cons,
		tol=optim_tol,
		method='SLSQP',
		options={
			'maxiter': 100000,
			'disp': True
		}
	)
	is_failed = (not sol.success)
	if is_failed:
		raise RuntimeError('Minimization Failed.')

	return restore(sol.x, m, d)
