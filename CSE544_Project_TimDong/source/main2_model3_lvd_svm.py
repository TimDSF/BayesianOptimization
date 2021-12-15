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
from gpytorch.kernels import MaternKernel
from gpytorch.kernels import ScaleKernel
from gpytorch.constraints import GreaterThan
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from torch.optim import SGD

from util import colorbar

# func = lambda x: np.log(np.log((4 - 2.1 * np.square(x[0]) + 1/3 * np.power(x[0], 4)) * np.square(x[0]) + x[0] * x[1] + 4 * (-1 + np.square(x[1])) * np.square(x[1]) + 1.0279 + 0.5) + 0.6931 + 0.5)

# sampler = qmc.Sobol(d = 2, scramble = False)
# train_X = sampler.random_base2(m = 5)
# train_X = np.matmul(train_X, np.array([[6, 0], [0, 4]])) - np.array([3, 2])
# train_Y = np.array([func(x) for x in train_X]).reshape(-1, 1)

data = np.array([l.split(',')[0:4] for l in open('./data/svm.csv', 'r').read().split('\n')]).astype(float)

sampler = qmc.Sobol(d = 1, scramble = False)
sample = (sampler.random_base2(m = 5) * 288).reshape(-1).astype(int)

train_X = torch.tensor(data[sample, 0:3])
train_Y = torch.log(torch.tensor(data[sample, 3:4]) - 0.2)

def search(train_X, train_Y, kernel):
	GP = SingleTaskGP(train_X, train_Y)
	GP.covar_module = kernel
	GP.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-12))
	GP.likelihood.noise_covar.raw_noise.requires_grad_(False)
	GP.likelihood.noise = 1e-12

	mll = ExactMarginalLogLikelihood(GP.likelihood, GP)
	mll = mll.to(train_X)

	print('BEFORE:')
	print('  Constant Mean:', GP.mean_module.constant.item())
	print('  BIC:', np.log(32)*3 - 2 * mll(GP(train_X), GP.train_targets).item())

	mll = fit_gpytorch_model(mll)

	print('AFTER:')
	print('  Constant Mean:', GP.mean_module.constant.item())
	print('  BIC:', np.log(32)*3 - 2 * mll(GP(train_X), GP.train_targets).item())
	print()

search(train_X, train_Y, RBFKernel())
search(train_X, train_Y, MaternKernel(0.5))
search(train_X, train_Y, MaternKernel(1.5))
search(train_X, train_Y, MaternKernel(2.5))