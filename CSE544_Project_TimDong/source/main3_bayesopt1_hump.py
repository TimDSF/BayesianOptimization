import numpy as np
from scipy.stats import qmc
from scipy.stats import norm
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

## 0
func = lambda x: np.log(np.log((4 - 2.1 * np.square(x[0]) + 1/3 * np.power(x[0], 4)) * np.square(x[0]) + x[0] * x[1] + 4 * (-1 + np.square(x[1])) * np.square(x[1]) + 1.0279 + 0.5) + 0.6931 + 0.5)

sampler = qmc.Sobol(d = 2, scramble = False)
train_X = torch.tensor(np.matmul(sampler.random_base2(m = 5), np.array([[6, 0], [0, 4]])) - np.array([3, 2]))
train_Y = - torch.tensor(np.array([func(x) for x in train_X]).reshape(-1, 1))

GP = SingleTaskGP(train_X, train_Y)
GP.covar_module = RBFKernel()
GP.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-12))
GP.likelihood.noise_covar.raw_noise.requires_grad_(False)
GP.likelihood.noise = 1e-12

mll = ExactMarginalLogLikelihood(GP.likelihood, GP)
mll = mll.to(train_X)
mll = fit_gpytorch_model(mll)

N = 1000
N_1 = N + 1
bounds = torch.tensor([[-3, -2], [3, 2]])
X = torch.linspace(-3, 3, N_1)
Y = - torch.linspace(-2, 2, N_1)
grid = torch.tensor([[x, y] for y in Y for x in X])
XX, YY = torch.meshgrid(X, Y)
posterior = GP.posterior(grid, observation_noise = False)

## 1
# mu: mean; sigma: standard deviation; phi_: current optimal
def EI(mu, sigma, phi_):
	return (mu - phi_) * norm.cdf((mu - phi_) / sigma) + sigma * norm.pdf((mu - phi_) / sigma)

mean = posterior.mean.reshape(N_1, N_1).detach().numpy()
sd = torch.sqrt(posterior.variance.reshape(N_1, N_1)).detach().numpy()
ei = EI(mean, sd, np.max(train_Y.numpy()))
y_, x_  = np.unravel_index(np.argmax(ei), ei.shape); x_ = x_ / N * 6 - 3; y_ = 2 - y_ / N * 4

## 2
colorbar(plt.imshow(mean, interpolation = 'nearest', extent = [-3, 3, -2, 2]))
plt.show()

colorbar(plt.imshow(sd, interpolation = 'nearest', extent = [-3, 3, -2, 2]))
plt.show()

colorbar(plt.imshow(ei, interpolation = 'nearest', extent = [-3, 3, -2, 2]))
plt.scatter(x_, y_, marker = 'x')
plt.show()