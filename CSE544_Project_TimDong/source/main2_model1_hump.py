import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from sklearn.neighbors import KernelDensity

import math
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel
from gpytorch.kernels import ScaleKernel
from gpytorch.constraints import GreaterThan
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from torch.optim import SGD

from util import colorbar

## 1
func = lambda x: (4 - 2.1 * np.square(x[0]) + 1/3 * np.power(x[0], 4)) * np.square(x[0]) + x[0] * x[1] + 4 * (-1 + np.square(x[1])) * np.square(x[1])

sampler = qmc.Sobol(d = 2, scramble = False)
train_X = sampler.random_base2(m = 5)
train_X = np.matmul(train_X, np.array([[6, 0], [0, 4]])) - np.array([3, 2])
train_Y = np.array([func(x) for x in train_X]).reshape(-1, 1)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# for i in range(len(train_Y)):
# 	ax.plot([train_X[i, 0], train_X[i, 0]], [train_X[i, 1], train_X[i, 1]], [0, train_Y[i]])

# ax.scatter(train_X[:, 0], train_X[:, 1], 0, marker = 'x')
# ax.scatter(train_X[:, 0], train_X[:, 1], train_Y, marker = '^')
# plt.show()

## 2
train_X = torch.tensor(train_X)
train_Y = torch.tensor(train_Y)
GP = SingleTaskGP(train_X, train_Y)
# GP.covar_module = RBFKernel() # default Matern_2.5
GP.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-12))
GP.likelihood.noise_covar.raw_noise.requires_grad_(False)
GP.likelihood.noise = 1e-12

mll = ExactMarginalLogLikelihood(GP.likelihood, GP)
mll = mll.to(train_X)

print('BEFORE:')
print('  Constant Mean:', GP.mean_module.constant.item())
print('  Lengthscale:', GP.covar_module.base_kernel.lengthscale.tolist())
print('  Output Scale:', GP.covar_module.outputscale.item())
print('  MLL:', mll(GP(train_X), GP.train_targets).item())

mll = fit_gpytorch_model(mll)

print('AFTER:')
print('  Constant Mean:', GP.mean_module.constant.item())
print('  Lengthscale:', GP.covar_module.base_kernel.lengthscale.tolist())
print('  Output Scale:', GP.covar_module.outputscale.item())
print('  MLL:', mll(GP(train_X), GP.train_targets).item())

## 4

N = 101
bounds = torch.tensor([[-3, -2], [3, 2]])
X = torch.linspace(-3, 3, N)
Y = torch.linspace(2, -2, N)
grid = torch.tensor([[x, y] for y in Y for x in X])
XX, YY = torch.meshgrid(X, Y)
posterior = GP.posterior(grid, observation_noise = False)
mean = posterior.mean.reshape(N, N)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(XX.detach().numpy(), YY.detach().numpy(), mean.detach().numpy())
# plt.show()

colorbar(plt.imshow(mean.detach().numpy(), interpolation = 'nearest', extent = [-3, 3, -2, 2]))
plt.show()


def fun1(x, y):
	return (4 - 2.1 * np.square(x) + 1/3 * np.power(x, 4)) * np.square(x) + x * y + 4 * (-1 + np.square(y)) * np.square(y)

z = fun1(X[None, :], Y[:, None])

res = np.abs(mean.detach().numpy() - z.numpy())
colorbar(plt.imshow(res, interpolation = 'nearest', extent = [-3, 3, -2, 2]))
plt.show()

res = res.reshape(-1)
plt.hist(res, bins = 80, range = (-80, 80))
plt.show()

## 5
sd = torch.sqrt(posterior.variance.reshape(N, N))
colorbar(plt.imshow(sd.detach().numpy(), interpolation = 'nearest', extent = [-3, 3, -2, 2]))
plt.show()

## 6
# kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.5).fit(res.reshape(-1, 1))
# grid = np.linspace(-10, 100, 5001).reshape(-1, 1)
# plt.plot(grid, np.exp(kde.score_samples(grid)))
# plt.show()
