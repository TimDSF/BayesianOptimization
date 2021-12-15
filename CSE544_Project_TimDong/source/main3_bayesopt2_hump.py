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

N = 1000
N_1 = N + 1
X = torch.linspace(-3, 3, N_1)
Y = torch.linspace(2, -2, N_1)
bounds = torch.tensor([[-3.0, -2.0], [3.0, 2.0]])
grid = torch.tensor([[x, y] for y in Y for x in X])

# max: -0.7082094417989129
def bayesopt_hump(m_start, m_policy):
	sampler = qmc.Sobol(d = 2)

	train_X = torch.tensor(np.matmul(sampler.random(m_start), np.array([[6, 0], [0, 4]])) - np.array([3, 2]))
	train_Y = torch.tensor(np.array([func(x[0], x[1]) for x in train_X]).reshape(-1, 1))

	sobol_X = torch.tensor(np.matmul(sampler.random(m_start + m_policy), np.array([[6, 0], [0, 4]])) - np.array([3, 2]))
	sobol_Y = torch.tensor(np.array([func(x[0], x[1]) for x in sobol_X]).reshape(-1, 1))

	rand_Y = torch.tensor([[func(random.random() * 6 - 3, random.random() * 4 - 2)] for _ in range(m_policy)])
	rand_Y = torch.cat((train_Y, rand_Y), 0)

	GP = SingleTaskGP(train_X, train_Y)
	GP.covar_module = RBFKernel()
	GP.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
	GP.likelihood.noise_covar.raw_noise.requires_grad_(False)
	GP.likelihood.noise = 1e-5

	# mll = ExactMarginalLogLikelihood(GP.likelihood, GP)
	# mll = mll.to(train_X)
	# mll = fit_gpytorch_model(mll)

	for i in range(m_policy):
		# EI = ExpectedImprovement(GP, best_f = torch.max(train_Y))
		# candidate, acq = optimize_acqf(EI, bounds = bounds, q = 1, num_restarts = 10, raw_samples = 10)
		# observation = torch.tensor([[func(candidate[0, 0], candidate[0, 1])]])
		# GP = GP.condition_on_observations(candidate, observation)

		posterior = GP.posterior(grid, observation_noise = False)
		mean = posterior.mean.reshape(N_1, N_1).detach().numpy()
		sd = torch.sqrt(posterior.variance.reshape(N_1, N_1)).detach().numpy()
		ei = EI(mean, sd, np.max(train_Y.numpy()))

		y_, x_ = np.unravel_index(np.argmax(ei), ei.shape)
		x_ = x_ / N * 6 - 3; y_ = 2 - y_ / N * 4
		# print(i, x_, y_, func(x_, y_), np.max(ei))

		candidate = torch.tensor([[x_, y_]])
		observation = torch.tensor([[func(x_, y_)]])

		# colorbar(plt.imshow(mean, interpolation = 'nearest', extent = [-3, 3, -2, 2]))
		# for x in train_X:
		# 	plt.scatter(x[0], x[1], c = 'g')
		# plt.show()

		# colorbar(plt.imshow(sd, interpolation = 'nearest', extent = [-3, 3, -2, 2]))
		# for x in train_X:
		# 	plt.scatter(x[0], x[1], c = 'g')
		# plt.show()

		# colorbar(plt.imshow(ei, interpolation = 'nearest', extent = [-3, 3, -2, 2]))
		# for x in train_X:
		# 	plt.scatter(x[0], x[1], c = 'g')
		# plt.scatter(x_, y_, marker = 'x')
		# plt.show()

		GP = GP.condition_on_observations(candidate, observation, noise = torch.tensor([[1e-5]]))

		# train_X = torch.cat((train_X, torch.tensor([[x_, y_]])), 0)
		train_Y = torch.cat((train_Y, observation), 0)

		# plt.scatter(train_X[:, 0], train_X[:, 1], s = 5)
		# plt.show()

	return train_Y, sobol_Y, rand_Y

n_reps = 20
m_start = 5
m_policy = 500

train_Y = torch.zeros((n_reps, m_start + m_policy, 1))
sobol_Y = torch.zeros((n_reps, m_start + m_policy, 1))
rand_Y  = torch.zeros((n_reps, m_start + m_policy, 1))

for i in range(n_reps):
	print(i)
	train_Y[i], sobol_Y[i], rand_Y[i] = bayesopt_hump(m_start, m_policy)

train_Y = train_Y.numpy()
sobol_Y = sobol_Y.numpy()
rand_Y  = rand_Y.numpy()

bayop_gaps = np.array([np.divide(np.amax(train_Y[:, 0:m_start+1+i], 1) - np.amax(train_Y[:, 0:m_start], 1), 0.7082094417989129 - np.amax(train_Y[:, 0:m_start], 1)) for i in range(m_policy)])
sobol_gaps = np.array([np.divide(np.amax(sobol_Y[:, 0:m_start+1+i], 1) - np.amax(sobol_Y[:, 0:m_start], 1), 0.7082094417989129 - np.amax(sobol_Y[:, 0:m_start], 1)) for i in range(m_policy)])
rand_gaps  = np.array([np.divide(np.amax( rand_Y[:, 0:m_start+1+i], 1) - np.amax( rand_Y[:, 0:m_start], 1), 0.7082094417989129 - np.amax( rand_Y[:, 0:m_start], 1)) for i in range(m_policy)])

bayop_gap = np.mean(bayop_gaps, axis = 1)
sobol_gap = np.mean(sobol_gaps, axis = 1)
rand_gap  = np.mean( rand_gaps, axis = 1)

plt.plot(bayop_gap, label = 'BayesOpt')
plt.plot(sobol_gap, label = 'Sobol')
plt.plot( rand_gap, label = 'Random')
plt.legend()
plt.show()

p = np.array([ttest_ind(bayop_gaps[29, :, 0], rand_gaps[i, :, 0], equal_var = False, alternative = 'greater') for i in range(m_policy)])
plt.plot(p[:, 1], label = 'p-value (1-side)')
plt.legend()
plt.show()

print(bayop_gap[29, 0], bayop_gap[59, 0], bayop_gap[89, 0], bayop_gap[119, 0], bayop_gap[149, 0])
print(sobol_gap[29, 0], sobol_gap[59, 0], sobol_gap[89, 0], sobol_gap[119, 0], sobol_gap[149, 0])
print(p[29, 1], p[59, 1], p[89, 1], p[119, 1], p[149, 1])