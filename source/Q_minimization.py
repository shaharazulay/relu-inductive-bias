import numpy as np
from scipy.optimize import minimize


def flatten(w, a):
	return np.hstack([w.flatten(), a.flatten()])


def restore(v, m, d):
	w = v[: m * d].reshape(m, d)
	a = v[m * d:].reshape(m,)
	return w, a


def q_func(x, u_p_0, v_p_0, u_n_0, v_n_0):
	f = 0
	for i in range(len(x)):
		k = 2 * u_p_0[i] ** 2 + 2 * v_p_0[i] ** 2
		f += (k/4) * (1 - np.sqrt(1 + 4*x[i]**2/k**2) + (2*x[i]/k) * np.arcsinh(2*x[i]/k))
	return f


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


def solver(x, y, u_p_0, v_p_0, u_n_0, v_n_0, obj='L1', optim_tol=1e-3, x_0=None):
	x0 = (np.multiply(u_p_0, v_p_0) - np.multiply(u_n_0, v_n_0)).reshape(-1,)
	if x_0 is not None:
		x0 = x_0

	cons = {'type': 'ineq', 'fun': lambda v: optim_tol - np.abs(np.matmul(v.reshape(-1, 1).transpose(), x) - y).reshape(-1,)}

	if obj == 'L1':
		objective = lambda v: np.linalg.norm(v, ord=1)
	elif obj == 'L2':
		objective = lambda v: np.linalg.norm(v, ord=2)
	elif obj == 'Q':
		objective = lambda v: q_func(v.reshape(-1, 1), u_p_0, v_p_0, u_n_0, v_n_0)
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

	return sol.x
