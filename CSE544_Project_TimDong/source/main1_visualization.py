import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from sklearn.neighbors import KernelDensity

from util import colorbar
## 1
def main1():
	def fun1(x, y):
		return (4 - 2.1 * np.square(x) + 1/3 * np.power(x, 4)) * np.square(x) + x * y + 4 * (-1 + np.square(y)) * np.square(y)

	x = np.linspace(-3, 3, 1000)
	y = -np.linspace(-2, 2, 1000)

	z = fun1(x[None, :], y[:, None])
	
	colorbar(plt.imshow(z, interpolation = 'nearest', extent = [-3, 3, -2, 2]))
	plt.show()

## 3
def main3():
	def fun2(x, y):
		return np.log(np.log((4 - 2.1 * np.square(x) + 1/3 * np.power(x, 4)) * np.square(x) + x * y + 4 * (-1 + np.square(y)) * np.square(y) + 1.0279 + 0.5) + 0.6931 + 0.5)

	x = np.linspace(-3, 3, 1000)
	y = -np.linspace(-2, 2, 1000)

	z = fun2(x[None, :], y[:, None])

	colorbar(plt.imshow(z, interpolation = 'nearest', extent = [-3, 3, -2, 2]))
	plt.show()

## 4
def main4():
	data = np.array([l.split(',')[0:4] for l in open('./data/lda.csv', 'r').read().split('\n')]).astype(float)
	kde = KernelDensity(kernel = 'gaussian', bandwidth = 100).fit(data[:, 3:4])
	grid = np.linspace(0, 6000, 6001).reshape(-1, 1)
	plt.plot(grid, np.exp(kde.score_samples(grid)))
	plt.show()

	data = np.array([l.split(',')[0:4] for l in open('./data/svm.csv', 'r').read().split('\n')]).astype(float)
	kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.01).fit(data[:, 3:4])
	grid = np.linspace(0, 1, 5001).reshape(-1, 1)
	plt.plot(grid, np.exp(kde.score_samples(grid)))
	plt.show()

## 5	
def main5():
	data = np.array([l.split(',')[0:4] for l in open('./data/lda.csv', 'r').read().split('\n')]).astype(float)
	data[:, 3] = np.log(data[:, 3] - 1250)
	kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.2).fit(data[:, 3:4])
	grid = np.linspace(0, 10, 5001).reshape(-1, 1)
	plt.plot(grid, np.exp(kde.score_samples(grid)))
	plt.show()

	data = np.array([l.split(',')[0:4] for l in open('./data/svm.csv', 'r').read().split('\n')]).astype(float)
	data[:, 3] = np.log(data[:, 3] - 0.2)
	kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.1).fit(data[:, 3:4])
	grid = np.linspace(-4, -1, 5001).reshape(-1, 1)
	plt.plot(grid, np.exp(kde.score_samples(grid)))
	plt.show()

## main
main1()
main3()
