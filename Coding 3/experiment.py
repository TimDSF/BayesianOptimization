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

import matplotlib.pyplot as plt

def plotGP(GP, EI, grid, X, Y, rsample, candidate = None, title = 'figure'):
	acqf = EI(grid.reshape(-1, 1, 1))

	posterior = GP.posterior(grid, observation_noise = False)
	lower, upper = posterior.mvn.confidence_region()

	fig, axs = plt.subplots(2, figsize = (16, 4))
	
	axs[0].plot(grid, rsample.tolist(), c = 'red')
	axs[0].fill_between(grid.tolist(), lower.tolist(), upper.tolist(), alpha = 0.5)
	axs[0].plot(grid, posterior.mean.tolist())
	axs[0].scatter(X.tolist(), Y.tolist(), marker = '.', c = 'black', zorder = 5)

	axs[1].plot(grid.tolist(), acqf.tolist(), c = 'green')
	axs[1].fill_between(grid.tolist(), acqf.tolist(), color = 'green', alpha = 0.2)
	if candidate:
		axs[1].scatter(candidate.tolist(), 0, marker = '^', c = 'black')
	
	plt.savefig('figures/' + title)
	# plt.show()

X = torch.tensor([[18.018018018018], [22.7327327327327], [10.2402402402402]])
Y = torch.tensor([[-0.764057972761234], [1.37212149409023], [1.02368358590807]])

GP = SingleTaskGP(X, Y)
GP.covar_module = RBFKernel()
mll = ExactMarginalLogLikelihood(GP.likelihood, GP)
GP.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
GP.likelihood.noise_covar.raw_noise.requires_grad_(False)
GP.likelihood.noise = 1e-4
fit_gpytorch_model(mll)

bounds = torch.tensor([[0.0], [30.0]])

N = 2001
M = 100
grid = torch.linspace(0, 30, N)
posterior = GP.posterior(grid, observation_noise = False)
rsample = posterior.rsample().reshape(-1)

for i in range(3, M):
	EI = ExpectedImprovement(GP, best_f = torch.max(Y))
	candidate, acq_value = optimize_acqf(EI, bounds = bounds, q = 1, num_restarts = 10, raw_samples = 20)

	plotGP(GP, EI, grid, X, Y, rsample, candidate, str(i))
	print(i, candidate, acq_value)

	x = candidate
	idx = (candidate / 30 * N).int().item()
	idx = idx if idx != N else N-1
	y = rsample[idx].reshape(1, 1)
	X = torch.cat((X, x))
	Y = torch.cat((Y, y))

	GP = GP.condition_on_observations(x, y)
	
plotGP(GP, EI, grid, X, Y, rsample, None, str(M))
