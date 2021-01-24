import numpy as np
from scipy.optimize import minimize


def flatten(u, v):
	return np.hstack([u.flatten(), v.flatten()])


def restore(nu, d):
	u = nu[: d].reshape(d,)
	v = nu[d:].reshape(d,)
	return u, v


def q_func(x, s=0):
	return x * np.log(x + np.sqrt(x**2 + s**2)) - np.sqrt(x**2 + s**2)


def Q_func(u, v, mu, s=0):
	"""
	:param u: matrix of shape d representing the first layer u
	:param v: vector of shape d representing the second layer v
	:param mu: vector of shape dx 1 for each coordinate
	"""
	assert len(mu) == len(s)
	assert len(v) == len(mu)

	cnt = 0
	f = 0
	for u_i, v_i, mu_i, s_i in zip(u, v, mu, s):
		cnt += 1
		f += mu_i * q_func((u_i * v_i) / mu_i, s=s_i)
	assert cnt == len(mu)
	return f


def Q_func_objective(nu, mu, s, d):
	u, v = restore(nu, d)
	return Q_func(u, v, mu, s)


def margin_constraint(nu, x, y, d):
	"""
	for all n, y_n * sum((u * v) x_n) >= 1
	"""
	u, v = restore(nu, d)

	y_pred = np.dot(np.multiply(u, v).transpose(), x.transpose())
	margins = np.multiply(y, y_pred)

	return margins - 1


def calc_w_tilde(u, v):
	return np.multiply(u, v)


def solver(x, y, u_0, v_0, d, obj='L1', mu=None, s=None, optim_tol=1e-6, x0=None):
	nu_0 = flatten(u_0, v_0)
	if x0 is None:
		x0 = nu_0
	cons = {'type': 'ineq', 'fun': lambda v: margin_constraint(v, x, y, d)}

	if obj == 'L1':
		objective = lambda nu: np.linalg.norm(nu, ord=2)
	elif obj == 'Q':
		objective = lambda nu: Q_func_objective(nu, mu, s, d)
	else:
		raise ValueError('objective not supported.')

	sol = minimize(
		fun=objective,
		x0=x0,
		constraints=cons,
		tol=optim_tol,
		method='SLSQP',
		options={
			'maxiter': 100000,
			'disp': False
		}
	)
	is_failed = (not sol.success)
	if is_failed:
		raise RuntimeError('Minimization Failed.')

	return restore(sol.x, d)
