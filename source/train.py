import numpy as np
from tqdm import tqdm

from classifier import update, current_training_loss


def train(u_p_0, v_p_0, u_n_0, v_n_0, x, y, step_size, n_epochs, eval_freq=1000, early_stop=1e-4):
	training_loss = []
	u_p_array = []
	v_p_array = []
	u_n_array = []
	v_n_array = []

	u_p, v_p, u_n, v_n = np.array(u_p_0), np.array(v_p_0), np.array(u_n_0), np.array(v_n_0)

	for epoch in tqdm(range(n_epochs)):

		u_p_updated, v_p_updated, u_n_updated, v_n_updated = update(u_p, v_p, u_n, v_n, x, y, step_size)

		if (epoch + 1) % eval_freq == 0:
			# store learned weights
			training_loss.append(current_training_loss(u_p, v_p, u_n, v_n, x, y))
			u_p_array.append(u_p.copy())
			v_p_array.append(v_p.copy())
			u_n_array.append(u_n.copy())
			v_n_array.append(v_n.copy())

			if training_loss[-1] < early_stop:
				print(f'early stop at epoch {epoch + 1}...')
				break

		u_p, v_p, u_n, v_n = u_p_updated, v_p_updated, u_n_updated, v_n_updated

	return {
		'u_p': u_p_array,
		'v_p': v_p_array,
		'u_n': u_n_array,
		'v_n': v_n_array,
		'training_loss': training_loss
	}
