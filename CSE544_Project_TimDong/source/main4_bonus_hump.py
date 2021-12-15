import numpy as np
import seaborn as sns
from scipy.stats import qmc
from scipy.stats import norm
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from sklearn.neighbors import KernelDensity

import math
import torch
import random
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

import warnings
warnings.filterwarnings("ignore")

random.seed(544)

## 0
def func(x, y):
	return - np.log(np.log((4 - 2.1 * np.square(x) + 1/3 * np.power(x, 4)) * np.square(x) + x * y + 4 * (-1 + np.square(y)) * np.square(y) + 1.0279 + 0.5) + 0.6931 + 0.5)

def EI(mu, sigma, phi_):
	return (mu - phi_) * norm.cdf((mu - phi_) / sigma) + sigma * norm.pdf((mu - phi_) / sigma)

N = 100
N_1 = N + 1
X = torch.linspace(-3, 3, N_1)
Y = torch.linspace(2, -2, N_1)
bounds = torch.tensor([[-3.0, -2.0], [3.0, 2.0]])
grid = torch.tensor([[x, y] for y in Y for x in X])

# max: -0.7082094417989129
def bayesopt_hump(m_start, m_policy):
	sampler = qmc.Sobol(d = 2)

	train_X = torch.tensor(np.matmul(sampler.random(m_start), np.array([[6, 0], [0, 4]])) - np.array([3, 2]))
	train1_Y = torch.tensor(np.array([func(x[0], x[1]) for x in train_X]).reshape(-1, 1))
	train2_Y = torch.clone(train1_Y)
	train3_Y = torch.clone(train1_Y)

	random_Y = torch.tensor([[func(random.random() * 6 - 3, random.random() * 4 - 2)] for _ in range(m_policy)])
	random_Y = torch.cat((train1_Y, random_Y), 0)

	GP1 = SingleTaskGP(train_X, train1_Y)
	GP1.covar_module = RBFKernel()
	GP1.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-8))
	GP1.likelihood.noise_covar.raw_noise.requires_grad_(False)
	GP1.likelihood.noise = 1e-8

	GP2 = SingleTaskGP(train_X, train2_Y)
	GP2.covar_module = RBFKernel()
	GP2.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-8))
	GP2.likelihood.noise_covar.raw_noise.requires_grad_(False)
	GP2.likelihood.noise = 1e-8

	GP3 = SingleTaskGP(train_X, train3_Y)
	GP3.covar_module = RBFKernel()
	GP3.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-8))
	GP3.likelihood.noise_covar.raw_noise.requires_grad_(False)
	GP3.likelihood.noise = 1e-8

	# mll1 = ExactMarginalLogLikelihood(GP1.likelihood, GP1)
	# mll1 = mll1.to(train_X)
	# mll1 = fit_gpytorch_model(mll1)

	mll2 = ExactMarginalLogLikelihood(GP2.likelihood, GP2)
	mll2 = mll2.to(train_X)
	mll2 = fit_gpytorch_model(mll2)

	for i in range(m_policy):
		posterior = GP1.posterior(grid, observation_noise = False)
		mean = posterior.mean.reshape(N_1, N_1).detach().numpy()
		sd = torch.sqrt(posterior.variance.reshape(N_1, N_1)).detach().numpy()
		ei = EI(mean, sd, np.max(train1_Y.numpy()))
		y_, x_ = np.unravel_index(np.argmax(ei), ei.shape)
		x_ = x_ / N * 6 - 3; y_ = 2 - y_ / N * 4
		candidate = torch.tensor([[x_, y_]])
		observation = torch.tensor([[func(x_, y_)]])
		train1_Y = torch.cat((train1_Y, observation), 0)
		GP1 = GP1.condition_on_observations(candidate, observation, noise = torch.tensor([[1e-8]]))

		posterior = GP2.posterior(grid, observation_noise = False)
		mean = posterior.mean.reshape(N_1, N_1).detach().numpy()
		sd = torch.sqrt(posterior.variance.reshape(N_1, N_1)).detach().numpy()
		ei = EI(mean, sd, np.max(train2_Y.numpy()))
		y_, x_ = np.unravel_index(np.argmax(ei), ei.shape)
		x_ = x_ / N * 6 - 3; y_ = 2 - y_ / N * 4
		candidate = torch.tensor([[x_, y_]])
		observation = torch.tensor([[func(x_, y_)]])
		train2_Y = torch.cat((train2_Y, observation), 0)
		GP2 = GP2.condition_on_observations(candidate, observation, noise = torch.tensor([[1e-8]]))

		mll3 = ExactMarginalLogLikelihood(GP3.likelihood, GP3)
		mll3 = mll3.to(train_X)
		mll3 = fit_gpytorch_model(mll3)

		posterior = GP3.posterior(grid, observation_noise = False)
		mean = posterior.mean.reshape(N_1, N_1).detach().numpy()
		sd = torch.sqrt(posterior.variance.reshape(N_1, N_1)).detach().numpy()
		ei = EI(mean, sd, np.max(train3_Y.numpy()))
		y_, x_ = np.unravel_index(np.argmax(ei), ei.shape)
		x_ = x_ / N * 6 - 3; y_ = 2 - y_ / N * 4
		candidate = torch.tensor([[x_, y_]])
		observation = torch.tensor([[func(x_, y_)]])
		train_X = torch.cat((train_X, candidate), 0)
		train3_Y = torch.cat((train3_Y, observation), 0)
		GP3 = GP3.condition_on_observations(candidate, observation, noise = torch.tensor([[1e-8]]))

	return train1_Y, train2_Y, train3_Y, random_Y

n_reps = 20
m_start = 5
m_policy = 150

train1_Y = torch.zeros((n_reps, m_start + m_policy, 1))
train2_Y = torch.zeros((n_reps, m_start + m_policy, 1))
train3_Y = torch.zeros((n_reps, m_start + m_policy, 1))
random_Y = torch.zeros((n_reps, m_start + m_policy, 1))

for i in range(n_reps):
	print(i)
	train1_Y[i], train2_Y[i], train3_Y[i], random_Y[i] = bayesopt_hump(m_start, m_policy)

train1_Y = train1_Y.numpy()
train2_Y = train2_Y.numpy()
train3_Y = train3_Y.numpy()
random_Y = random_Y.numpy()

train1_gaps = np.array([np.divide(np.amax(train1_Y[:, 0:m_start+1+i], 1) - np.amax(train1_Y[:, 0:m_start], 1), 0.7082094417989129 - np.amax(train1_Y[:, 0:m_start], 1)) for i in range(m_policy)])
train2_gaps = np.array([np.divide(np.amax(train2_Y[:, 0:m_start+1+i], 1) - np.amax(train2_Y[:, 0:m_start], 1), 0.7082094417989129 - np.amax(train2_Y[:, 0:m_start], 1)) for i in range(m_policy)])
train3_gaps = np.array([np.divide(np.amax(train3_Y[:, 0:m_start+1+i], 1) - np.amax(train3_Y[:, 0:m_start], 1), 0.7082094417989129 - np.amax(train3_Y[:, 0:m_start], 1)) for i in range(m_policy)])
random_gaps = np.array([np.divide(np.amax(random_Y[:, 0:m_start+1+i], 1) - np.amax(random_Y[:, 0:m_start], 1), 0.7082094417989129 - np.amax(random_Y[:, 0:m_start], 1)) for i in range(m_policy)])

train1_gap = np.mean(train1_gaps, axis = 1)
train2_gap = np.mean(train2_gaps, axis = 1)
train3_gap = np.mean(train3_gaps, axis = 1)
random_gap = np.mean(random_gaps, axis = 1)

plt.plot(train1_gap, label = 'BayesOpt1')
plt.plot(train2_gap, label = 'BayesOpt2')
plt.plot(train3_gap, label = 'BayesOpt3')
plt.plot(random_gap, label = 'Random')
plt.legend()
plt.show()

p = np.array([ttest_ind(train3_gaps[i, :, 0], train1_gaps[i, :, 0], equal_var = False, alternative = 'greater') for i in range(m_policy)])
plt.plot(p[:, 1], label = 'p-value (1-side)')
plt.legend()
plt.show()

