import numpy as np
from tqdm import tqdm

from ntk import neural_tangent_kernel, kernel_distance
from classifier import update, minimal_margin, current_training_loss


def evaluate(u, v, x, y):
	"""
	calculate model accuracy (% of samples above a margin of 1) over dataset (x, y)
	"""
	y_pred = np.dot(np.multiply(u, v).transpose(), x.transpose())
	return np.sum(np.multiply(y_pred, y) > 1) / len(x)


def train(u_0, v_0, x, y, step_size, n_epochs, eval_freq=1000):
	training_loss = []
	training_accuracy = []
	minimal_margins = []
	u_array = []
	v_array = []

	kernel_distances = []
	#k_0 = lambda x, x_tag: neural_tangent_kernel(v_0, v_0, x, x_tag)

	u, v = np.array(u_0), np.array(v_0)

	for epoch in tqdm(range(n_epochs)):

		u_updated, v_updated, gamma_tilde, gamma = update(u, v, x, y, step_size)

		if (epoch + 1) % eval_freq == 0:
			minimal_margins.append(minimal_margin(u, v, x, y))
			training_loss.append(current_training_loss(u, v, x, y))
			training_accuracy.append(evaluate(u, v, x, y))
			u_array.append(u.copy())
			v_array.append(v.copy())

			# # calculate kernel distance
			# k_t = lambda x, x_tag: neural_tangent_kernel(w, a, x, x_tag)
			# kernel_distances.append(kernel_distance(k_t(x, x), k_0(x, x)))

		u, v = u_updated, v_updated

	return {
		'u': u_array,
		'v': v_array,
		'minimal_margin': minimal_margins,
		'training_loss': training_loss,
		'training_accuracy': training_accuracy,
		'kernel_distances': kernel_distances
	}
