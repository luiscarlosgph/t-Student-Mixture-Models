#!/usr/bin/env python
#
# @brief  Example that shows how to generate a random sample of data coming from a mixture of t-Student 
#         distributions and then use the 'smm.SMM' class to estimate the parameters of the original model 
#         using Expectation-Maximization.
#
# @author Luis C. Garcia-Peraza Herrera (luis.herrera.14@ucl.ac.uk).
# @date   20 Jul 2017.

import numpy as np

# My imports
import smm

def main():

	# Sample properties 
	n_dim = 3
	n_samples = 10000
	
	# Initialisation of random state
	seed = 1
	np.random.seed(seed)

	# Create random mixture of t-Student distributions
	n_components = 2
	max_mean = 100
	max_covar = 100
	max_df = 10
	means = np.random.randint(max_mean, size = (n_components, n_dim)) 
	vector_covars = np.random.randint(max_covar, size = (n_components, int((n_dim + 1) * n_dim / 2)))
	covars = np.zeros((n_components, n_dim, n_dim))
	for c in range(n_components):
		covars[c][np.triu_indices(n_dim)] = vector_covars[c]
		covars[c] = covars[c].T + covars[c] + 1.0
		min_eig = np.min(np.real(np.linalg.eigvals(covars[c])))
		if min_eig < 0:
			covars[c] -= 10 * min_eig * np.eye(*covars[c].shape)
	dfs = np.random.randint(max_df, size = (n_components,)) + 1.0
	weights = np.random.rand(n_components)
	weights /= weights.sum()

	# Generate random observations of the previous mixture 
	obs = []
	for c in range(n_components):
		c_obs = smm.SMM.multivariate_t_rvs(means[c], covars[c], df=dfs[c], n=n_samples)
		c_obs = c_obs[0:int(np.round(n_samples * weights[c]))]
		obs.append(c_obs)
	obs = np.vstack(obs)
	
	# Fit mixture of t-Students to data
	mix_t = smm.SMM(n_components=n_components, covariance_type='full', random_state=seed, tol=1e-12, 
		min_covar=1e-6, n_iter=1000, n_init=1, params='wmcd', init_params='wmcd')
	mix_t.fit(obs)

	# Show results
	print('==================================================')
	print('t-Student mixture optimized vs real parameters:')
	print('')
	print('   Number of samples: '              + str(n_samples))
	print('   Component[s] in the mixture: '    + str(n_components))
	print('')
	print('   Estimated mean[s]:\n'             + str(mix_t.means))
	print('   Correct mean[s]:\n'               + str(means))
	print('')
	print('   Estimated covariance[s]:\n'       + str(mix_t.covariances)) 
	print('   Correct covariance[s]:\n'         + str(covars))
	print('')
	print('   Estimated degree[s] of freedom: ' + str(mix_t.degrees))
	print('   Correct degree[s] of freedom: '   + str(dfs))
	print('')
	print('   Estimated weight[s]: '            + str(mix_t.weights))
	print('   Correct weight[s]: '              + str(weights))
	print('')
	print('==================================================')

if __name__ == '__main__':
	main()
