import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm



# init function for the weights (with specific shape and scale)
def init(alpha, s, m, d, seed=None):
	"""
	alpha = |a_0| + ||w_0||
	s = (|a_0| - ||w_0||) / (|a_0| + ||w_0||)
	"""
	if seed:
		np.random.seed(seed)

	norms_w = []
	norms_a = []
	for alpha_i, s_i in zip(alpha, s):
		norm_w = np.sqrt(alpha_i * (1 - s_i) / (1 + s_i))
		norm_a = np.sqrt(alpha_i * (1 + s_i) / (1 - s_i))
		norms_w.append(norm_w)
		norms_a.append(norm_a)

	w_0 = np.random.normal(size=(m, d), loc=0, scale=1)
	w_0_norms = np.linalg.norm(w_0, axis=1, ord=2)
	w_0 = w_0 / w_0_norms[:, np.newaxis] * np.array(norms_w)[:, np.newaxis]

	a_0 = np.random.normal(size=(1, m), loc=0, scale=1)
	a_0 = np.multiply(np.ones_like(a_0) * np.array(norms_a), (1 * (a_0 > 0) - 0.5) * 2)

	return w_0, a_0


def neural_tangent_kernel(w, a, x, x_tag):

	def gradient(w, a, z):
		activations = np.maximum(np.dot(w, z.transpose()), 0)
		c_i = 1.0 * (activations > 0)

		w_grad = np.multiply(np.dot(c_i, z), a.transpose())
		a_grad = activations.transpose()
		return np.hstack([w_grad.flatten(), a_grad.flatten()])

	grad_x = np.array(list(map(lambda x: gradient(w, a, x.reshape(1, -1)), x)))
	grad_x_tag = np.array(list(map(lambda x: gradient(w, a, x.reshape(1, -1)), x_tag)))
	return np.dot(grad_x, grad_x_tag.transpose())


def fit_svm_with_tangent_kernel(w, a, x, y):
	kernel = lambda x, x_tag: neural_tangent_kernel(w, a, x, x_tag)
	clf = svm.SVC(kernel=kernel)
	clf.fit(x, y)
	return clf


def plot_svm_classifier(clf, x, y):
	xmin = -1
	xmax = 1

	mesh_step = 0.02
	_x1 = np.arange(xmin, xmax, mesh_step)
	_x2 = np.arange(xmin, xmax, mesh_step)
	xx_1, xx_2 = np.meshgrid(_x1, _x2)

	input_ = np.c_[np.ones(xx_1.ravel().shape), xx_1.ravel(), xx_2.ravel()]
	y_pred = clf.predict(input_)

	plt.figure()
	z = np.reshape(np.sign(y_pred), xx_1.shape)
	plt.pcolormesh(xx_1, xx_2, z)

	plt.scatter([x_[1] for x_, y_ in zip(x, y) if y_ > 0], [x_[2] for x_, y_ in zip(x, y) if y_ > 0])
	plt.scatter([x_[1] for x_, y_ in zip(x, y) if y_ < 0], [x_[2] for x_, y_ in zip(x, y) if y_ < 0])
	plt.show()


if __name__ == '__main__':
	# define dimensions
	n = 100
	m = 20
	d = 3

	# define dataset
	x = np.array([[1, -0.8, -0.5], [1, -0.4, -0.5], [1, 0.15, 0.3]])
	y = np.array([[1.0, -1.0, 1.0]]).reshape(-1,)

	# init the weights
	alpha = [10000] * m
	s = [0] * m

	w_0, a_0 = init(alpha=alpha, s=s,  m=m, d=d)

	clf = fit_svm_with_tangent_kernel(w_0, a_0, x, y)
	plot_svm_classifier(clf, x, y)