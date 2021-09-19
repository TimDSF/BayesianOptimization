import os
import numpy as np
import matplotlib.pyplot as plt

def graph(title, x, y, f_mean, f_var, sx, sy, acq):
	f_hi = f_mean + 1.96 * f_var
	f_lo = f_mean - 1.96 * f_var
	lim = max(np.max(np.abs(y)) * 1.2, 2.5)

	ax1 = plt.subplot(211)
	ax1.set_title('Posterior Distribution after '+str(len(sx))+' samples')
	f1 = ax1.plot(x, y, color = 'red', alpha = 0.6, zorder = 2)
	f2 = ax1.plot(x, f_mean, color = 'blue', alpha = 0.6, zorder = 3)
	f3 = ax1.scatter(sx, sy, 3, color = 'black', zorder = 4)
	f4 = ax1.fill_between(x, f_hi, f_lo, color = 'blue', alpha = 0.2, zorder = 1)
	ax1.set_ylim([lim, -lim])

	ax2 = plt.subplot(212, sharex = ax1)
	ax2.plot(x, acq, zorder = 2)
	ax2.fill_between(x, acq, color = 'green', alpha = 0.2, zorder = 1)
	# ax2.set_ylim([0, 2])

	# plt.show()
	plt.savefig(title+'/'+title+'_'+str(len(sx)))
	plt.close()

	# fig = plt.figure(figsize=(12,4))
	# axes = fig.add_axes([0.1,0.1,0.8,0.8])
	# lim = max(np.max(np.abs(y)) * 1.2, 2.5)
	# plt.ylim(top = lim, bottom = -lim)

	# axes.plot(x, y, zorder = 2)
	# axes.plot(x, f_mean, zorder = 3)
	# axes.scatter(sx, sy, 3, color = 'black', zorder = 4)
	# axes.fill_between(x, f_hi, f_lo, color = 'blue', alpha = 0.2, zorder = 1)
	# plt.show()
	# plt.close()

def process(title, le = 0, ri = 25, n = 250, le_svar = 0, ri_svar = 0.5, num_samples = 50):
	# parameters
	svar = np.linspace(le_svar, ri_svar, n)
	 
	# function values
	x = np.linspace(le, ri, n)
	u = np.zeros(n)
	K = np.exp(-np.square(np.tile(x, (n, 1)) - np.tile(np.transpose([x]), (1, n))))
	y = np.random.seed(544)
	y = np.random.multivariate_normal(u, K)

	idx = np.array([], dtype = int)
	sx = np.array([])
	sy = np.array([])

	f_mean = np.zeros(n)
	f_cov = np.array(K)
	f_var = np.diag(f_cov)

	to_pick = np.array([True for _ in range(n)])

	# samples

	for i in range(num_samples):
		acq = np.exp(f_var ** 2) - 1
		graph(title, x, y, f_mean, f_var, sx, sy, acq)
		print(title + ': ' + str(i) + ' samples')
		
		j = np.random.choice(np.flatnonzero(acq == np.max(acq * to_pick)))
		to_pick[j] = False
		idx = np.append(idx, j)
		sx = np.append(sx, x[idx[i]])
		sy = np.append(sy, y[idx[i]] + np.random.normal(0, svar[j]))

		m = u[idx]
		k = K[idx]
		N = np.diag(np.ones(i+1) * svar[idx])
		C = K[:, idx][idx] + N

		_invC = np.linalg.inv(C)
		_dif = sy-m
		for j1 in range(n):
			f_mean[j1] = u[j1] + np.matmul(np.matmul(np.transpose(k[:, j1]), _invC), _dif)

			_prod1 = np.matmul(np.transpose(k[:, j1]), _invC)
			for j2 in range(n):
				f_cov[j1][j2] = K[j1][j2] - np.matmul(_prod1, k[:, j2])

		f_var = np.diag(f_cov)

	acq = np.exp(f_var ** 2) - 1
	graph(title, x, y, f_mean, f_var, sx, sy, acq)
	print(title + ': ' + str(num_samples) + ' samples')


# os.mkdir('Exact')
# os.mkdir('Small')
# os.mkdir('Large')
# os.mkdir('Linear')
# os.mkdir('LinearExtreme')

# process('Exact', le_svar = 0, ri_svar = 0)
# process('Small', le_svar = 0.3, ri_svar = 0.3)
# process('Large', le_svar = 1, ri_svar = 1)
# process('Linear', le_svar = 0, ri_svar = 1)
# process('LinearExtreme', le = 0, ri = 100, n = 1000, le_svar = 0, ri_svar = 1, num_samples = 250)