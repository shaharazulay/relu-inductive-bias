import numpy as np
from tqdm import tqdm

from ntk import neural_tangent_kernel, kernel_distance
from classifier import update, minimal_margin
from Q_minimization import calc_w_tilde_norms, solver


def evaluate(w, a, x, y):
	"""
	calculate model accuracy (% of samples above a margin of 1) over dataset (x, y)
	"""
	activations = np.maximum(np.dot(w, x.transpose()), 0)
	y_pred = np.dot(a, activations)
	return np.sum(np.multiply(y_pred, y) > 1) / len(x)


def train(w_0, a_0, x, y, m, d, alpha, s, step_size, n_epochs, eval_freq=1000, eval_freq_Q=10000):
	training_loss = []
	training_accuracy = []
	w_tilde_norms_array = []
	w_array = []
	a_array = []

	training_loss_Q = []
	w_tilde_norms_Q_array = []

	kernel_distances = []
	k_0 = lambda x, x_tag: neural_tangent_kernel(w_0, a_0, x, x_tag)

	w, a = np.array(w_0), np.array(a_0)

	for epoch in tqdm(range(n_epochs)):

		w, a, gamma_tilde, gamma = update(w, a, x, y, epoch, step_size)

		if (epoch + 1) % eval_freq == 0:
			# store learned weights and their norms
			w_tilde_norms = calc_w_tilde_norms(w, a) / gamma
			w_tilde_norms_array.append(w_tilde_norms)

			training_loss.append(gamma_tilde)
			training_accuracy.append(evaluate(w, a, x, y))
			w_array.append(w.copy())
			a_array.append(a.copy())

			# calculate kernel distance
			k_t = lambda x, x_tag: neural_tangent_kernel(w, a, x, x_tag)
			kernel_distances.append(kernel_distance(k_t(x, x), k_0(x, x)))

		# Q minimization
		mu = [2 * alpha_i / gamma_tilde for alpha_i in alpha]

		if ((epoch + 1) % eval_freq_Q == 0) or ((epoch + 1) % 1000 == 0) and (mu[0] > 10):
			if evaluate(w, a, x, y) < 1:  # didn't reach sufficient margin solution yet
				pass
			else:
				try:
					w_opt_Q, a_opt_Q = solver(
						x,
						y,
						w_0,
						a_0,
						m,
						d,
						obj='Q',
						mu=mu,
						s=s,
						x0=np.random.normal(size=(m * (d + 1),)),
						optim_tol=1e-10 if mu[0] > 10 else 1e-7
					)

					gamma = minimal_margin(w_opt_Q, a_opt_Q, x, y)
					w_tilde_norms_Q = calc_w_tilde_norms(w_opt_Q, a_opt_Q) / gamma
					w_tilde_norms_Q_array.append(w_tilde_norms_Q)
					training_loss_Q.append(gamma_tilde)
				except Exception as e:
					print(f'Mu = {mu[0]}:: {e}')

	return {
		'w': w_array,
		'a': a_array,
		'w_tilde_norms': w_tilde_norms_array,
		'training_loss': training_loss,
		'training_accuracy': training_accuracy,
		'training_loss_Q': training_loss_Q,
		'w_tilde_norms_Q': w_tilde_norms_Q_array,
		'kernel_distances': kernel_distances
	}
