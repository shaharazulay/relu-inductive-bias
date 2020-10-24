import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


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


def kernel_distance(k_t, k_0):
	return 1 - np.trace(np.dot(k_t, k_0.transpose())) / (np.linalg.norm(k_t, ord='fro') * np.linalg.norm(k_0, ord='fro'))


def fit_svm_with_tangent_kernel(w, a, x, y):
	kernel = lambda x, x_tag: neural_tangent_kernel(w, a, x, x_tag)
	clf = svm.SVC(C=np.inf, kernel=kernel)
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
	plt.pcolormesh(xx_1, xx_2, z, cmap='coolwarm')

	plt.scatter([x_[1] for x_, y_ in zip(x, y) if y_ > 0], [x_[2] for x_, y_ in zip(x, y) if y_ > 0], s=100, c='k', marker='+')
	plt.scatter([x_[1] for x_, y_ in zip(x, y) if y_ < 0], [x_[2] for x_, y_ in zip(x, y) if y_ < 0], s=100, c='k', marker='_')
	plt.show()
