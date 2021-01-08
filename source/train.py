import numpy as np
from tqdm import tqdm

from ntk import neural_tangent_kernel, kernel_distance
from classifier import update, minimal_margin, current_training_loss
from Q_minimization import calc_w_tilde_norms, solver


def evaluate(w, a, x, y):
	"""
	calculate model accuracy (% of samples above a margin of 1) over dataset (x, y)
	"""
	activations = np.maximum(np.dot(w, x.transpose()), 0)
	y_pred = np.dot(a, activations)
	return np.sum(np.multiply(y_pred, y) > 1) / len(x)


def train(w_0, a_0, x, y, step_size, n_epochs, eval_freq=1000):
	training_loss = []
	training_accuracy = []
	w_tilde_norms_array = []
	w_array = []
	a_array = []

	kernel_distances = []
	k_0 = lambda x, x_tag: neural_tangent_kernel(w_0, a_0, x, x_tag)

	w, a = np.array(w_0), np.array(a_0)

	for epoch in tqdm(range(n_epochs)):

		w_updated, a_updated, gamma_tilde, gamma = update(w, a, x, y, epoch, step_size)

		if (epoch + 1) % eval_freq == 0:
			# store learned weights and their norms
			w_tilde_norms = calc_w_tilde_norms(w, a) / minimal_margin(w, a, x, y)
			w_tilde_norms_array.append(w_tilde_norms)

			training_loss.append(current_training_loss(w, a, x, y))
			training_accuracy.append(evaluate(w, a, x, y))
			w_array.append(w.copy())
			a_array.append(a.copy())

			# calculate kernel distance
			k_t = lambda x, x_tag: neural_tangent_kernel(w, a, x, x_tag)
			kernel_distances.append(kernel_distance(k_t(x, x), k_0(x, x)))

		w, a = w_updated, a_updated

	return {
		'w': w_array,
		'a': a_array,
		'w_tilde_norms': w_tilde_norms_array,
		'training_loss': training_loss,
		'training_accuracy': training_accuracy,
		'kernel_distances': kernel_distances
	}
