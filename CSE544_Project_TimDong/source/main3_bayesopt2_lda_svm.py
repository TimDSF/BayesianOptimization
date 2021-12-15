import numpy as np
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
from gpytorch.kernels import MaternKernel
from gpytorch.kernels import ScaleKernel
from gpytorch.constraints import GreaterThan
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from torch.optim import SGD

from util import colorbar

import warnings
warnings.filterwarnings("ignore")

## 0
def EI(mu, sigma, phi_):
		return (mu - phi_) * norm.cdf((mu - phi_) / sigma) + sigma * norm.pdf((mu - phi_) / sigma)

prob = 'svm' # lda svm
shift = 1250 if prob == 'lda' else 0.2
data = torch.tensor(np.array(sorted([tuple(np.array(l.split(',')[0:4]).astype(float)) for l in open('./data/'+prob+'.csv', 'r').read().split('\n')])))

N = data.shape[0]
G0 = sorted(list(set(data[:, 0].tolist()))); N0 = len(G0)
G1 = sorted(list(set(data[:, 1].tolist()))); N1 = len(G1)
G2 = sorted(list(set(data[:, 2].tolist()))); N2 = len(G2)
G = torch.tensor([[G0[x], y, z] for x in range(N0) for y in range(N1) for z in range(N2)]) if prob == 'lda' else torch.tensor([[x, G1[y], z] for x in range(N0) for y in range(N1) for z in range(N2)])

M = data.shape[0]
data[:, 3] = - torch.log(data[:, 3] - shift)
max_ = torch.max(data[:, 3]).item()+0.01

mp = lambda x: x[0] * N1 * N2 + x[1] * N2 + x[2]

# max: -7.1436
# max:  3.1917
def bayesopt_lda_svm(m_start, m_policy):
	sampler = qmc.Sobol(d = 3)

	train_X = torch.tensor(np.matmul(sampler.random(m_start), np.array([[N0, 0, 0], [0, N1, 0], [0, 0, N2]])).astype(int))
	train_Y = torch.tensor([[data[mp(x), 3:4]] for x in train_X])
	train_X = train_X.type(torch.float64)
	if prob == 'lda':
		train_X[:, 0] = torch.tensor(G0)[train_X[:, 0].type(torch.int64)]
	else:
		train_X[:, 1] = torch.tensor(G1)[train_X[:, 1].type(torch.int64)]

	sobol_X = torch.tensor(np.matmul(sampler.random(m_start + m_policy), np.array([[N0, 0, 0], [0, N1, 0], [0, 0, N2]])).astype(int))
	sobol_Y = torch.tensor([[data[mp(x), 3:4]] for x in sobol_X])

	rand_Y = torch.tensor([[data[int(random.random() * M), 3]] for _ in range(m_start + m_policy)])

	GP = SingleTaskGP(train_X, train_Y)
	GP.covar_module = MaternKernel() # RBFKernel() MaternKernel()
	GP.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-6))
	GP.likelihood.noise_covar.raw_noise.requires_grad_(False)
	GP.likelihood.noise = 1e-6

	# mll = ExactMarginalLogLikelihood(GP.likelihood, GP)
	# mll = mll.to(train_X)
	# mll = fit_gpytorch_model(mll)

	for i in range(m_policy):
		posterior = GP.posterior(G, observation_noise = False)
		mean = posterior.mean.detach().numpy()
		sd = torch.sqrt(posterior.variance).detach().numpy()
		ei = EI(mean, sd, np.max(train_Y.numpy()))
		
		x_ = np.argmax(ei)
		candidate = G[x_].reshape(1, -1)
		if prob == 'lda':
			candidate[0, 0] = G0[int(candidate[0, 0].item())]
		else:
			candidate[0, 1] = G1[int(candidate[0, 1].item())]
		observation = data[x_, 3].reshape(1, 1)

		GP = GP.condition_on_observations(candidate, observation, noise = torch.tensor([[1e-6]]))
		train_Y = torch.cat((train_Y, observation), 0)

	return train_Y, sobol_Y, rand_Y

n_reps = 20
m_start = 5
m_policy = 500

train_Y = torch.zeros((n_reps, m_start + m_policy, 1))
sobol_Y = torch.zeros((n_reps, m_start + m_policy, 1))
rand_Y  = torch.zeros((n_reps, m_start + m_policy, 1))

for i in range(n_reps):
	print(i)
	train_Y[i], sobol_Y[i], rand_Y[i] = bayesopt_lda_svm(m_start, m_policy)

train_Y = train_Y.numpy()
sobol_Y = sobol_Y.numpy()
rand_Y  = rand_Y.numpy()

bayop_gaps = np.array([np.divide(np.amax(train_Y[:, 0:m_start+1+i], 1) - np.amax(train_Y[:, 0:m_start], 1), max_ - np.amax(train_Y[:, 0:m_start], 1)) for i in range(m_policy)])
sobol_gaps = np.array([np.divide(np.amax(sobol_Y[:, 0:m_start+1+i], 1) - np.amax(sobol_Y[:, 0:m_start], 1), max_ - np.amax(sobol_Y[:, 0:m_start], 1)) for i in range(m_policy)])
rand_gaps  = np.array([np.divide(np.amax( rand_Y[:, 0:m_start+1+i], 1) - np.amax( rand_Y[:, 0:m_start], 1), max_ - np.amax( rand_Y[:, 0:m_start], 1)) for i in range(m_policy)])

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