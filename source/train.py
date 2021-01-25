import numpy as np
from tqdm import tqdm

from classifier import update, current_training_loss, update_with_relu, current_training_loss_with_relu


def train(u_0, v_0, a_0, x, y, step_size, n_epochs, eval_freq=1000, relu=False):
	training_loss = []
	u_array = []
	v_array = []
	a_array = []

	u, v, a = np.array(u_0), np.array(v_0), np.array(a_0)

	for epoch in tqdm(range(n_epochs)):

		if relu:
			u_updated, v_updated, a_updated = update_with_relu(u, v, a, x, y, step_size)
		else:
			u_updated, v_updated, a_updated = update(u, v, a, x, y, step_size)

		if (epoch + 1) % eval_freq == 0:
			# store learned weights and their norms
			if relu:
				training_loss.append(current_training_loss_with_relu(u, v, a, x, y))
			else:
				training_loss.append(current_training_loss(u, v, a, x, y))
			u_array.append(u.copy())
			v_array.append(v.copy())
			a_array.append(a.copy())

		u, v, a = u_updated, v_updated, a_updated

	return {
		'u': u_array,
		'v': v_array,
		'a': a_array,
		'training_loss': training_loss
	}
