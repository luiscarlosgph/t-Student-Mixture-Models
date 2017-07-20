"""
@brief t-Student Mixture Models.

@details This class has reused code and comments from sklearn.mixture.gmm.

This class is the implementsthe following paper:
================================================
 
'Robust mixture modelling using the t distribution', D. Peel and G. J. McLachlan.
Published at: Statistics and Computing (2000) 10, 339-348.
 
This code has been done as part of my PhD at University College London under the
supervision of Prof. Sebastien Ourselin and Dr. Tom Vercauteren.

@author Luis Carlos Garcia-Peraza Herrera (luis.herrera.14@ucl.ac.uk).
@date   24 Nov 2015.
"""

import numpy as np
import sklearn
import sklearn.cluster
import sklearn.utils
import scipy.linalg
import scipy.special
import scipy.optimize

class dofMaximizationError(ValueError):
	def __init__(self, message):
		super(dofMaximizationError, self).__init__(message)

class SMM(sklearn.base.BaseEstimator):
	"""t-Student Mixture Model
 
	Representation of a t-Student mixture model probability distribution.
	This class allows for easy evaluation of, sampling from, and
	maximum-likelihood estimation of the parameters of an SMM distribution.

	Initializes parameters such that every mixture component has zero
	mean and identity covariance.

	Parameters
	----------
	n_components : int, optional
		Number of mixture components. Defaults to 1.

	covariance_type : string, optional
		String describing the type of covariance parameters to
		use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
		Defaults to 'full'.

	random_state: RandomState or an int seed (None by default)
		A random number generator instance

	min_covar : float, optional
		Floor on the diagonal of the covariance matrix to prevent
		overfitting.  Defaults to 1e-6.

	tol : float, optional
		Convergence threshold. EM iterations will stop when average
		gain in log-likelihood is below this threshold.  Defaults to 1e-6.

	n_iter : int, optional
		Number of EM iterations to perform. Defaults to 1000.

	n_init : int, optional
		Number of initializations to perform. The best result is kept.
		Defaults to 1.

	params : string, optional
		Controls which parameters are updated in the training
		process.  Can contain any combination of 'w' for weights,
		'm' for means, 'c' for covars, and 'd' for the degrees of freedom.  
		Defaults to 'wmcd'.

	init_params : string, optional
		Controls which parameters are updated in the initialization
		process.  Can contain any combination of 'w' for weights,
		'm' for means, 'c' for covars, and 'd' for the degrees of freedom.  
		Defaults to 'wmcd'.

	Attributes
	----------
	weights_ : array, shape (`n_components`,)
		This attribute stores the mixing weights for each mixture component.

	means_ : array, shape (`n_components`, `n_features`)
		Mean parameters for each mixture component.

	covars_ : array
		Covariance parameters for each mixture component.  The shape
		depends on `covariance_type`::

			(n_components, n_features)             if 'spherical',
			(n_features, n_features)               if 'tied',
			(n_components, n_features)             if 'diag',
			(n_components, n_features, n_features) if 'full'

	_converged : bool
		True when convergence was reached in fit(), False otherwise.

	"""	

	def __init__(self, n_components=1, covariance_type=None, random_state=None, tol=None, 
		min_covar=None, n_iter=None, n_init=None, params=None, init_params=None):

		# Correct those parameters that are None
		if covariance_type is None:
			covariance_type = 'full'
		if tol is None:
			tol = 1e-6
		if min_covar is None:
			min_covar = 1e-6
		if n_iter is None:
			n_iter = 1000
		if n_init is None:
			n_init = 1
		if params is None:
			params = 'wmcd'
		if init_params is None:
			init_params = 'wmcd'
	
		# Store the parameters as class attributes
		self._n_components = n_components
		self._covariance_type = covariance_type
		self._random_state = random_state
		self._tol = tol
		self._min_covar = min_covar
		self._n_iter = n_iter
		self._n_init = n_init
		self._params = params
		self._init_params = init_params

		#if covariance_type not in ['spherical', 'tied', 'diag', 'full']:
		#	covariance_type = 'full'
			# raise ValueError('Invalid value for covariance_type: %s' % covariance_type)

		#if n_init < 1:
		#	n_init = 1
		#	# raise ValueError('SMM estimation requires at least one run.')

		# self.weights_ = np.ones(self._n_components) / self._n_components

		# Flag to indicate exit status of fit() method: converged (True) or
		# n_iter reached (False)
		self._converged = False
	
	def _expectation_step(self, X):
		"""
		@brief   Performs the expectation step of the Expectation-Maximisation algorithm to fit a mixture
		         t-students to the data.
		@details This method uses the means, class-related weights, covariances and degrees of freedom
		         stored in the attributes of this class: self.means_, self.weights_, self.covars_, and
		         self.degrees_. This method also uses the 
		
		@param[in] X matrix with all the data points, each row represents a new data point and the 
		             columns represent each one of the features.
		"""

		# Sanity checks:
		#    - Check that the fit() method has been called before this one.
		#    - Convert input to 2d array, raise error on sparse matrices. Calls assert_all_finite by default.
		#    - Check that the the X array is not empty of samples. 
		#    - Check that the no. of features is equivalent to the no. of means that we have in self.
		sklearn.utils.validation.check_is_fitted(self, 'means_')
		X = sklearn.utils.validation.check_array(X, dtype = np.float64)
		if X.ndim == 1:
			X = X[:, np.newaxis]
		if X.size == 0:
			return np.array([]), np.empty((0, self._n_components))
		if X.shape[1] != self.means_.shape[1]:
			raise ValueError('[SMM._expectation_step] Error, the shape of X is not compatible with self.')
		
		# Initialisation of reponsibilities and weight of each point for the Gamma distribution
		n_samples, n_dim = X.shape
		responsibilities = np.ndarray(shape = (X.shape[0], self._n_components), dtype = np.float64) 
		gammaweights_ = np.ndarray(shape = (X.shape[0], self._n_components), dtype = np.float64) 

		# Calculate the probability of each point belonging to each t-Student distribution of the mixture
		pr_before_weighting = self._multivariate_t_student_density(X, self.means_, self.covars_, 
			self.degrees_, self._covariance_type, self._min_covar)
		pr = pr_before_weighting * self.weights_

		# Calculate the likelihood of each point
		likelihoods = pr.sum(axis = 1)
		
		# Update responsibilities
		responsibilities = pr / (likelihoods.reshape(likelihoods.shape[0], 1) + 10 * SMM._EPS)

		# Update the Gamma weight for each observation
		mahalanobis_distance_mix_func = SMM._mahalanobis_funcs[self._covariance_type]
		gammaweights_ = (self.degrees_ + n_dim) / (self.degrees_ + mahalanobis_distance_mix_func(X, self.means_, self.covars_, self._min_covar))

		return likelihoods, responsibilities, gammaweights_
	
	def _maximisation_step(self, X, responsibilities, gammaweights_):
		"""
		@brief Perform the maximisation step of the EM algorithm.
	
		@param[in] X                List of k_features-dimensional data points. Each row corresponds 
	   	                         to a single data point. It is an array_like, with shape (n, n_features).
		@param[in] responsibilities matrix of responsibilities with the rows representing the points
	   	                         and the columns representing each class in the mixture.
		@param[in] gammaweights_    matrix of point weights where each row represents a point and each
	   	                         column represents a class in the mixture.
		"""

		n_samples, n_dim = X.shape
		z_sum = responsibilities.sum(axis = 0)
		zu = responsibilities * gammaweights_
		zu_sum = zu.sum(axis = 0)

		# Update weights
		if 'w' in self._params:
			self.weights_ = z_sum / n_samples

		# Update means
		if 'm' in self._params:
			self.means_ = np.dot(zu.T, X) / (zu_sum.reshape(zu_sum.shape[0], 1) + 10 * SMM._EPS)

		# Update covariances
		if 'c' in self._params:
			covar_mstep_func = SMM._covar_mstep_funcs[self._covariance_type]
			self.covars_ = covar_mstep_func(X, zu, z_sum, self.means_, self._min_covar)

		# Update degrees of freedom
		if 'd' in self._params:
			try:
				self.degrees_ = SMM._solve_dof_equation(self.degrees_, responsibilities, z_sum, gammaweights_, n_dim, self._tol, self._n_iter)
			except FloatingPointError, e:
				raise dofMaximizationError(e.message)
	
	def fit(self, X, y=None):
		"""
		@brief   Estimate model parameters with the expectation-maximization algorithm.
		@details A initialization step is performed before entering the em
		         algorithm. If you want to avoid this step, set the keyword
	   	      argument init_params to the empty string '' when creating the
	      	   SMM object. Likewise, if you would like just to do an
	         	initialization, set n_iter=0.
	
		@param[in] X List of k_features-dimensional data points. Each row corresponds 
	   	          to a single data point. It is an array_like, with shape (n, n_features).
		"""

		# Sanity check: assert that the input matrix is not 1D
		if (len(X.shape) == 1):
			raise ValueError('The input matrix must have a shape of (n_samples, n_features).')

		# Sanity checks:
		#    - Convert input to 2d array, raise error on sparse matrices. Calls assert_all_finite by default.
		#    - No. of samples is higher or equal to the no. of components in the mixture.
		X = sklearn.utils.validation.check_array(X, dtype = np.float64)
		if X.shape[0] < self._n_components:
			raise ValueError('SMM estimation with %s components, but got only %s samples' % (self._n_components, X.shape[0]))
		
		# For all the initialisations we get the one with the best parameters
		n_samples, n_dim = X.shape
		max_prob = -np.infty
		for _ in range(self._n_init):
			
			# EM initialisation
			if 'm' in self._init_params or not hasattr(self, 'means_'):
				self.means_ = sklearn.cluster.KMeans(n_clusters = self._n_components, init='k-means++', random_state = self._random_state).fit(X).cluster_centers_

			if 'w' in self._init_params or not hasattr(self, 'weights_'):
				self.weights_ = np.tile(1.0 / self._n_components, self._n_components)

			if 'c' in self._init_params or not hasattr(self, 'covars_'):
				cv = np.cov(X.T) + self._min_covar * np.eye(X.shape[1])
				if not cv.shape:
					cv.shape = (1, 1)
				self.covars_ = SMM.distribute_covar_matrix_to_match_covariance_type(cv, self._covariance_type, self._n_components)

			if 'd' in self._init_params or not hasattr(self, 'degrees_'):
				self.degrees_ = np.tile(1.0, self._n_components) 

			tol = self._tol
			self._converged = False
			current_log_likelihood = None

			# EM algorithm 
			for i in xrange(self._n_iter):
				prev_log_likelihood = current_log_likelihood

				# Expectation step
				likelihoods, responsibilities, gammaweights_ = self._expectation_step(X)

				# Sanity check: assert that the likelihoods, responsibilities and gammaweights have the correct
				#               dimensions
				assert(len(likelihoods.shape) == 1)
				assert(likelihoods.shape[0] == n_samples)
				assert(len(responsibilities.shape) == 2)
				assert(responsibilities.shape[0] == n_samples)
				assert(responsibilities.shape[1] == self._n_components)
				assert(len(gammaweights_.shape) == 2)
				assert(gammaweights_.shape[0] == n_samples)
				assert(gammaweights_.shape[1] == self._n_components)

				# Calculate loss function
				current_log_likelihood = np.log(likelihoods).mean()
			
				# Check for convergence
				if prev_log_likelihood is not None:
					change = np.abs(current_log_likelihood - prev_log_likelihood)
					if change < tol:
						self._converged = True
						break
				
				# Maximisation step
				try:
					self._maximisation_step(X, responsibilities, gammaweights_)
				except dofMaximizationError, e:
					print('[self._maximisation_step] Error in the maximization step of the degrees of ' + \
						'freedom: ' + e.message)
					break
			
			# If the results are better, keep it
			if self._n_iter and self._converged:
				if current_log_likelihood > max_prob:
					max_prob = current_log_likelihood
					best_params = {
						'weights': self.weights_,
						'means': self.means_,
						'covars': self.covars_,
						'degrees': self.degrees_
					}

		# Check the existence of an init param that was not subject to
		# likelihood computation issue
		if np.isneginf(max_prob) and self._n_iter:
			raise RuntimeError(
				'EM algorithm was never able to compute a valid likelihood ' +
				'given initial parameters. Try different init parameters ' +
				'(or increasing n_init) or check for degenerate data.')
		
		# Choosing the best result of all the iterations as the actual result
		if self._n_iter:
			self.weights_ = best_params['weights']
			self.means_ = best_params['means']
			self.covars_ = best_params['covars']
			self.degrees_ = best_params['degrees']
		return self
		
	def predict(self, X):
		"""
		@brief Predict label for data.
	
		@param[in] X Array-like, shape = [n_samples, n_features].
	
		@returns an array, shape = (n_samples,).
		"""
		likelihoods, responsibilities, gammaweights_ = self._expectation_step(X)
		return responsibilities.argmax(axis = 1)

	def predict_proba(self, X):
		"""
		@brief Predict label for data.
	
		@param[in] X Array-like, shape = [n_samples, n_features].
	
		@returns an array, shape = (n_samples,).
		"""
		
		likelihoods, responsibilities, gammaweights_ = self._expectation_step(X)
		return responsibilities
	
	def score(self, X, y=None):
		"""
		@brief Compute the log probability under the model.

		@param[in] X  Array_like, shape (n_samples, n_features). List of n_features-dimensional data points. 
		              Each row corresponds to a single data point.

		@returns prob Array_like, shape (n_samples,). Probabilities of each data point in X.
		"""
		
		prob, _ = self.score_samples(X)
		return prob
	
	def score_samples(self, X):
		"""
		@brief Per-sample likelihood of the data under the model.

		@details Compute the probability of X under the model and return the posterior distribution 
		         (responsibilities) of each mixture component for each element of X.

		@param[in] X array_like, shape (n_samples, n_features). List of n_features-dimensional data points. 
		             Each row corresponds to a single data point.

		@returns a prob, responsibilities.
                 prob             : array_like, shape (n_samples,). Probability of each data point in X.
		           responsibilities : array_like, shape (n_samples, n_components). Posterior probabilities of
		                              each mixture component for each observation.
		"""

		sklearn.utils.validation.check_is_fitted(self, 'means_')
		X = sklearn.utils.validation.check_array(X, dtype = np.float64)

		if X.ndim == 1:
			X = X[:, np.newaxis]
		if X.size == 0:
			return np.array([]), np.empty((0, self.n_components))
		if X.shape[1] != self.means_.shape[1]:
			raise ValueError('[score_samples] ValueError, the shape of X  is not compatible with self.')

		# Initialisation of reponsibilities and weight of each point for the Gamma distribution
		n_samples, n_dim = X.shape
		responsibilities = np.ndarray(shape = (X.shape[0], self._n_components), dtype = np.float64) 
		gammaweights_ = np.ndarray(shape = (X.shape[0], self._n_components), dtype = np.float64) 

		# Calculate the probability of each point belonging to each t-Student distribution of the mixture
		pr = self._multivariate_t_student_density(X, self.means_, self.covars_, self.degrees_, 
			self._covariance_type, self._min_covar) * self.weights_

		# Calculate the likelihood of each point
		likelihoods = pr.sum(axis = 1)
		
		# Update responsibilities
		responsibilities = pr / (likelihoods.reshape(likelihoods.shape[0], 1) + 10 * SMM._EPS)

		return likelihoods, responsibilities

	def _n_parameters(self):
		"""
		@returns the number of free parameters in the model.
		"""

		_, n_features = self.means_.shape
		if self._covariance_type == 'full':
			cov_params = self.n_components * n_features * (n_features + 1) / 2.
		elif self._covariance_type == 'diag':
			cov_params = self.n_components * n_features
		elif self._covariance_type == 'tied':
			cov_params = n_features * (n_features + 1) / 2.
		elif self._covariance_type == 'spherical':
			cov_params = self.n_components
		mean_params = n_features * self.n_components
		df_params = self.n_components
		return int(cov_params + mean_params + df_params + self.n_components - 1)	
	
	def bic(self, X):
		"""
		@brief Bayesian information criterion for the current model fit and the proposed data.
	
		@param[in] X array of data points with shape (n_samples, n_dimensions).
	 
		@returns a float (the lower the better).
		"""
		return (-2 * self.score(X).sum() + self._n_parameters() * np.log(X.shape[0]))
	
	def aic(self, X):	
		"""
		@brief Akaike information criterion for the current model fit and the proposed data.
		
		@param[in] X array of data points with shape (n_samples, n_dimensions).
	 
		@returns a float (the lower the better).
		"""
		return - 2 * self.score(X).sum() + 2 * self._n_parameters()
	
	@staticmethod
	def _solve_dof_equation(v_vector, z, z_sum, u, n_dim, tol, n_iter):
		"""
		@brief   Solves the equation to calculate the next value of v (degrees of freedom).
		@details This method is part of the maximisation step. It is used to calculate the next value
					for the degrees of freedom of each t-Student component. This method uses the
					information from the E-step as well as the number of dimensions (features) of a point.
		
		@param[in] v_vector Degrees of freedoom of ALL the components of the mixture.
		@param[in] z        Matrix of responsibilities, each row represents a point and each column
								  represents a component of the mixture.
		@param[in] z_sum    Sum of all the rows of the matrix of responsibilities.
		@param[in] u        Matrix of gamma weights, each row represents a point and each column
								  represents a component of the mixture.
		@param[in] n_dim    Number of features of each data point.
		
		@returns a vector with the updated degrees of freedom for each component in the mixture.
		"""
		
		# Sanity check: check that the dimensionality of the vector of degrees of freedom is correct
		assert(len(v_vector.shape) == 1)
		n_components = v_vector.shape[0]

		# Sanity check: the matrix of responsibilities should be (n_samples x n_components)
		assert(len(z.shape) == 2)
		assert(z.shape[1] == n_components)

		# Sanity check: the top-to-bottom sum of the responsibilities must have a shape (n_components, )
		assert(len(z_sum.shape) == 1)
		assert(z_sum.shape[0] == n_components)

		# Sanity check: gamma weights matrix must have the same dimensionality as the responsibilities
		assert(u.shape == z.shape)

		# Initialisation
		new_v_vector = np.empty_like(v_vector)	

		# Calculate the constant part of the equation to calculate the new degrees of freedom
		vdim = (v_vector + n_dim) / 2.0 
		zlogu_sum = np.sum(z * (np.log(u) - u), axis = 0) 
		constant_part = 1.0 + zlogu_sum / z_sum + scipy.special.digamma(vdim) - np.log(vdim)

		# Solve the equation numerically using Newton-Raphson for each component of the mixture
		for c in range(n_components):
			func = lambda x: np.log(x / 2.0) - scipy.special.digamma(x / 2.0) + constant_part[c]
			fprime = lambda x: 1.0 / x - scipy.special.polygamma(1, x / 2.0) / 2.0
			fprime2 = lambda x: - 1.0 / (x * x) - scipy.special.polygamma(2, x / 2.0) / 4.0
			new_v_vector[c] = scipy.optimize.newton(func, v_vector[c], fprime, args=(), tol = tol, 
				maxiter = n_iter, fprime2 = fprime2)
			if new_v_vector[c] < 0.0:
				raise ValueError('[_solve_dof_equation] Error, degree of freedom smaller than one. \n' + \
					'n_components[c] = ' + str(n_components) + '. \n' + \
					'v_vector[c] = ' + str(v_vector[c]) + '. \n' + \
					'new_v_vector[c] = ' + str(new_v_vector[c]) + '. \n' + \
					'constant_part[c] = ' + str(constant_part[c]) + '. \n' + \
					'zlogu_sum[c] = ' + str(zlogu_sum[c]) + '. \n' + \
					'z_sum[c] = ' + str(z_sum[c]) + '. \n' + \
					'z = ' + str(z) + '. \n')

		return new_v_vector
	
	@staticmethod
	def _covar_mstep_diag(X, zu, z_sum, means, min_covar):
		"""
		@brief Performing the covariance maximisation step for full covariances.

		@param[in] X         Array_like, shape (n_samples, n_features).
		                     List of k_features-dimensional data points. Each row corresponds to a single 
									data point.
									
		@param[in] zu        Array, shape (n_samples, n_components).
		                     Contains the element-wise multiplication of the responsibilities by the gamma 
									weights.

		@param[in] z_sum     Array, shape (n_components, )
		                     Sum of all the responsibilities for each mixture.

		@param[in] means     Array_like, shape (n_components, n_features).
									List of n_features-dimensional mean vectors for n_components t-Students.
									Each row corresponds to a single mean vector.

		@param[in] min_covar Float value.
		                     Minimum amount that will be added to the covariance matrix in case of trouble, 
									usually 1.e-6.
		"""

		# Eq. 15 from K. Murphy, "Fitting a Conditional Linear Gaussian Distribution" adapted to the 
		# mixture of t-students (i.e. responsibilities matrix is multiplied element-wise by the gamma 
		# weights matrix. See that zu.T is used in the calculation of weighted_X_sum)
		norm = np.float64(1) / (z_sum[:, np.newaxis] + 10 * SMM._EPS)
		weighted_X_sum = np.dot(zu.T, X) 
		avg_X2 = np.dot(zu.T, X * X) * norm
		avgmeans_2 = means ** 2
		avg_Xmeans_ = means * weighted_X_sum * norm
		return avg_X2 - 2 * avg_Xmeans_ + avgmeans_2 + min_covar

	@staticmethod
	def _covar_mstep_spherical(*args):
		cv = SMM._covar_mstep_diag(*args)
		return np.tile(cv.mean(axis = 1)[:, np.newaxis], (1, cv.shape[1]))
	
	@staticmethod
	def _covar_mstep_full(X, zu, z_sum, means, min_covar):
		"""
		@brief Performing the covariance maximisation step for full covariances.

		@param[in] X         Array_like, shape (n_samples, n_features).
		                     List of k_features-dimensional data points. Each row corresponds to a single 
									data point.
									
		@param[in] zu        Array, shape (n_samples, n_components).
		                     Contains the element-wise multiplication of the responsibilities by the gamma 
									weights.

		@param[in] z_sum     Array, shape (n_components, )
		                     Sum of all the responsibilities for each mixture.

		@param[in] means     Array_like, shape (n_components, n_features).
									List of n_features-dimensional mean vectors for n_components t-Students.
									Each row corresponds to a single mean vector.

		@param[in] min_covar Float value.
		                     Minimum amount that will be added to the covariance matrix in case of trouble, 
									usually 1.e-7.
		"""
		
		# Sanity checks for dimensionality
		n_samples, n_features = X.shape
		n_components = means.shape[0]
		assert(zu.shape[0] == n_samples)
		assert(zu.shape[1] == n_components)
		assert(z_sum.shape[0] == n_components)

		# Eq. 31 from D. Peel and G. J. McLachlan, "Robust mixture modelling using the t distribution"
		# n_components = zu.shape[1] 
		cv = np.empty((n_components, n_features, n_features))
		zu_sum = zu.sum(axis = 0)
		for c in range(n_components):
			post = zu[:, c]
			mu = means[c]
			diff = X - mu
			with np.errstate(under = 'ignore'):
				# Underflow Errors in doing post * X.T are not important
				if n_components == 1:
					avg_cv = np.dot(post * diff.T, diff) / (zu_sum[c] + 10 * SMM._EPS)
				else:
					avg_cv = np.dot(post * diff.T, diff) / (z_sum[c] + 10 * SMM._EPS)
				
			cv[c] = avg_cv + min_covar * np.eye(n_features)
		return cv

	@staticmethod
	def _covar_mstep_tied(X, zu, z_sum, means, min_covar):
		"""
		@brief Performing the covariance maximisation step for full covariances.

		@param[in] X         Array_like, shape (n_samples, n_features).
		                     List of k_features-dimensional data points. Each row corresponds to a single 
									data point.
									
		@param[in] zu        Array, shape (n_samples, n_components).
		                     Contains the element-wise multiplication of the responsibilities by the gamma 
									weights.

		@param[in] z_sum     Array, shape (n_components, )
		                     Sum of all the responsibilities for each mixture.

		@param[in] means     Array_like, shape (n_components, n_features).
									List of n_features-dimensional mean vectors for n_components t-Students.
									Each row corresponds to a single mean vector.

		@param[in] min_covar Float value.
		                     Minimum amount that will be added to the covariance matrix in case of trouble, 
									usually 1.e-7.
		"""

		avg_X2 = np.dot(zu.T, X * X)
		avg_means2 = np.dot(z_sum * means.T, means)
		out = avg_X2 - avg_means2
		out /= z_sum.sum()
		out.flat[::len(out) + 1] += min_covar
		return out

	@staticmethod
	def _multivariate_t_student_density_diag(X, means, covars, dfs, min_covar):
		"""
		@brief Multivariate t-Student PDF for a matrix of data points.
		
		@param[in] X         List of k_features-dimensional data points. Each row corresponds 
									to a single data point. It is an array_like, shape (n_samples, n_features).

		@param[in] means     is an array_like, shape (n_components, n_features).
									List of n_features-dimensional mean vectors for n_components t-Students.
									Each row corresponds to a single mean vector.

		@param[in] covars    is an array_like list of n_components covariance parameters for each t-Student. 
									Shape (n_components, n_features).

		@param[in] dfs       Degrees of freedom, shape (n_components, ).

		@param[in] min_covar Minimum amount that will be added to the covariance matrix in case of 
									trouble, usually 1.e-6.

		@returns the evaluation of the multivariate probability density function for a t-Student distribution.
		"""
		
		# Sanity check: make sure that the shape of means and covariances is as expected for diagonal matrices
		n_samples, n_dim = X.shape
		assert(covars.shape[0] == means.shape[0])
		assert(covars.shape[1] == means.shape[1])
		assert(covars.shape[1] == n_dim)
	
		# Calculate inverse and determinant of the covariances
		inv_covars = 1.0 / covars
		det_covars = np.prod(covars, axis = 1)

		# Calculate the value of the numerator
		num = scipy.special.gamma((dfs + n_dim) / 2.0)
		
		# Calculate Mahalanobis distance from all the points to the mean of each component in the mix
		maha = SMM._mahalanobis_distance_mix_diag(X, means, covars, min_covar) 

		# Calculate the value of the denominator
		braces = 1.0 + maha / dfs
		denom = np.power(np.pi * dfs, n_dim / 2.0) * scipy.special.gamma(dfs / 2.0) *  np.sqrt(det_covars) * np.power(braces, (dfs + n_dim) / 2.0)

		return num / denom 
	
	@staticmethod
	def _multivariate_t_student_density_spherical(X, means, covars, dfs, min_covar):
		"""
		@brief Multivariate t-Student PDF for a matrix of data points.
		
		@param[in] X         List of k_features-dimensional data points. Each row corresponds 
									to a single data point. It is an array_like, shape (n_samples, n_features).

		@param[in] means     is an array_like, shape (n_components, n_features).
									List of n_features-dimensional mean vectors for n_components t-Students.
									Each row corresponds to a single mean vector.

		@param[in] covars    is an array_like list of n_components covariance parameters for each t-Student. 
									Shape (n_components, n_features).

		@param[in] dfs       Degrees of freedom, shape (n_components, ).

		@param[in] min_covar Minimum amount that will be added to the covariance matrix in case of 
									trouble, usually 1.e-6.

		@returns the evaluation of the multivariate probability density function for a t-Student distribution.
		"""

		cv = covars.copy()
		if covars.ndim == 1:
			cv = cv[:, np.newaxis]
		if covars.shape[1] == 1:
			cv = np.tile(cv, (1, X.shape[-1]))
		
		# Sanity check: make sure that the covariance is spherical, i.e. all the elements of the diagonal must
		#               be equal
		for k in range(cv.shape[0]):
			assert(np.unique(cv[k]).shape[0] == 1)

		return SMM._multivariate_t_student_density_diag(X, means, cv, dfs, min_covar)

	@staticmethod
	def _multivariate_t_student_density_tied(X, means, covars, dfs, min_covar):
		"""
		@brief Multivariate t-Student PDF for a matrix of data points.
		
		@param[in] X         List of k_features-dimensional data points. Each row corresponds 
									to a single data point. It is an array_like, shape (n_samples, n_features).

		@param[in] means     is an array_like, shape (n_components, n_features).
									List of n_features-dimensional mean vectors for n_components t-Students.
									Each row corresponds to a single mean vector.

		@param[in] covars    is an array_like list of n_components covariance parameters for each t-Student. 
									Shape (n_features, n_features).

		@param[in] dfs       Degrees of freedom, shape (n_components, ).

		@param[in] min_covar Minimum amount that will be added to the covariance matrix in case of 
									trouble, usually 1.e-6.

		@returns the evaluation of the multivariate probability density function for a t-Student distribution.
		"""
		
		# Sanity check: make sure that the shape is (n_features, n_features) and that it matches the shape
		#               of the vector of means
		assert(len(covars.shape) == 2)
		assert(covars.shape[0] == covars.shape[1])
		assert(means.shape[1] == covars.shape[0])

		cv = np.tile(covars, (means.shape[0], 1, 1))
		return SMM._multivariate_t_student_density_full(X, means, cv, dfs, min_covar)
	
	@staticmethod
	def _multivariate_t_student_density_full(X, means, covars, dfs, min_covar):
		n_samples, n_dim = X.shape
		n_components = len(means)
		prob = np.empty((n_samples, n_components))
		
		# Sanity check: assert that the received means and covars have the right shape
		assert(means.shape[0]  == n_components)
		assert(covars.shape[0] == n_components)
		assert(dfs.shape[0]    == n_components)

		# We evaluate all the saples for each component 'c' in the mixture
		for c, (mu, cv, df) in enumerate(zip(means, covars, dfs)):

			# Calculate the Cholesky decomposition of the covariance matrix
			cov_chol = SMM._cholesky(cv, min_covar)

			# Calculate the determinant of the covariance matrix
			cov_det = np.power(np.prod(np.diagonal(cov_chol)), 2)

			# Calculate the Mahalanobis distance between each vector and the mean
			maha = SMM._mahalanobis_distance_chol(X, mu, cov_chol)

			# Calculate the coefficient of the gamma functions
			r = np.asarray(df, dtype = np.float64)
			gamma_coef = np.exp(scipy.special.gammaln((r + n_dim) / 2.0) - scipy.special.gammaln(r / 2.0))

			# Calculate the denominator of the multivariate t-Student
			denom = np.sqrt(cov_det) * np.power(np.pi * df, n_dim / 2.0) * np.power((1 + maha / df), (df + n_dim) / 2)
			
			# Finally calculate the PDF of the class 'c' for all the X samples 
			prob[:, c] = gamma_coef / denom 
		
		return prob
	
	@staticmethod
	def _multivariate_t_student_density(X, means, covars, dfs, cov_type, min_covar):
		"""
		@brief   Calculates the PDF of the multivariate t-student for a group of samples.
		@details This method has a dictionary with the different types of covariance
					matrices and calls the appropriate PDF function depending on the type
					of covariance matrix that is passed as parameter.
					This method assumes that the covariance matrices are full if the
					parameter cov_type is not specified when calling.
		
		@param[in] X        is an array_like, shape (n_samples, n_features)
						        List of n_features-dimensional data points.  Each row corresponds to a
						        single data point.

		@param[in] means    is an array_like, shape (n_components, n_features)
							     List of n_features-dimensional mean vectors for n_components t-Students.
							     Each row corresponds to a single mean vector.

		@param[in] covars   is an array_like list of n_components covariance parameters for each t-Student. 
							     The shape depends on `covariance_type`:
							     (n_components, n_features)             if 'spherical',
								  (n_features, n_features)               if 'tied',
								  (n_components, n_features)             if 'diag',
								  (n_components, n_features, n_features) if 'full'

		@param[in] cov_type is a string that indicates the type of the covariance parameters.
								  It must be one of 'spherical', 'tied', 'diag', 'full'.  Defaults to 'full'.

		@returns an array_like, shape (n_samples, n_components) array containing the log probabilities 
					of each data point in X under each of the n_components multivariate t-Student distributions.
		"""

		_multivariate_normal_density_dict = {
			'diag': SMM._multivariate_t_student_density_diag,
			'spherical': SMM._multivariate_t_student_density_spherical,
			'tied': SMM._multivariate_t_student_density_tied,
			'full': SMM._multivariate_t_student_density_full
		}
		return _multivariate_normal_density_dict[cov_type](X, means, covars, dfs, min_covar)
		
	@staticmethod
	def _cholesky(cv, min_covar):
		"""
		@brief Calculates the lower triangular Cholesky decomposition of a covariance matrix.
		
		@param[in] covar     Covariance matrix whose Cholesky decomposition wants to be calculated.
		@param[in] min_covar Minimum amount that will be added to the covariance matrix in case of 
									trouble, usually 1.e-6.

		@returns the lower Cholesky decomposition of a covariance matrix, shape (n_features, n_features).
		"""
		
		# Sanity check: assert that the covariance matrix is squared 
		assert(cv.shape[0] == cv.shape[1])

		# Sanity check: assert that the covariance matrix is symmetric
		if (cv.transpose() - cv).sum() > min_covar:
			print('[SMM._cholesky] Error, covariance matrix not symmetric: ' + str(cv))
		
		n_dim = cv.shape[0]
		try:
			cov_chol = scipy.linalg.cholesky(cv, lower = True) 
		except scipy.linalg.LinAlgError:
			# The model is most probably stuck in a component with too
			# few observations, we need to reinitialize this components
			cov_chol = scipy.linalg.cholesky(cv + min_covar * np.eye(n_dim), lower = True)
		return cov_chol
	
	@staticmethod
	def _mahalanobis_distance_chol(X, mu, cov_chol):
		"""
		@brief Calculates the Mahalanobis distance between a matrix (set) of vectors (X) and another vector (mu).
		@details The vectors must be organised by row in X, that is, the features are the columns.

		@param[in] X               Matrix with a vector in each row.
		@param[in] mu              Mean vector of a single distribution (no mixture).
		@param[in] cov_chol        Cholesky decomposition (L, i.e. lower triangular) of the covariance 
								         (normalising) matrix in case that is a full matrix. 
											The shape is (n_features, n_features).

		@returns a vector of distances, each row represents the distance from the vector in the same row of X and mu. 
		"""
		z = scipy.linalg.solve_triangular(cov_chol, (X - mu).T, lower = True)
		return np.einsum('ij, ij->j', z, z)
		
	@staticmethod
	def _mahalanobis_distance_mix_diag(X, means, covars, min_covar):
		"""
		@brief Calculates the mahalanobis distance between a matrix of points and a mixture of distributions
		       when the covariance matrices are diagonal.
		
		@param[in] X      Matrix with a vector in each row.
		@param[in] means  is an array_like, shape (n_components, n_features)
								List of n_features-dimensional mean vectors for n_components t-Students.
								Each row corresponds to a single mean vector.
		@param[in] covars is an array_like list of n_components covariance parameters for each t-Student. 
								The shape depends is (n_components, n_features).
		
		@returns The Mahalanobis distance from all the samples to all the component means, 
					shape (n_samples, n_components).
		"""
		n_samples, n_dim = X.shape
		n_components = len(means)
		result = np.empty((n_samples, n_components))
		for c, (mu, cv) in enumerate(zip(means, covars)):
			centred_X = X - mu
			inv_cov = np.float64(1) / cv  
			result[:, c] = (centred_X * inv_cov * centred_X).sum(axis = 1)
		return result

	@staticmethod
	def _mahalanobis_distance_mix_spherical(*args):
		return SMM._mahalanobis_distance_mix_diag(*args)

	@staticmethod
	def _mahalanobis_distance_mix_full(X, means, covars, min_covar):
		"""
		@brief Calculates the mahalanobis distance between a matrix of points and a mixture of distributions. 
		
		@param[in] X      Matrix with a vector in each row.

		@param[in] means  is an array_like, shape (n_components, n_features)
								List of n_features-dimensional mean vectors for n_components t-Students.
								Each row corresponds to a single mean vector.

		@param[in] covars is an array_like list of n_components covariance parameters for each t-Student. 
								The shape is (n_components, n_features, n_features).
		
		@returns The Mahalanobis distance from all the samples to all the component means, 
					shape (n_samples, n_components).
		"""
		n_samples, n_dim = X.shape
		n_components = len(means)
		result = np.empty((n_samples, n_components))
		for c, (mu, cv) in enumerate(zip(means, covars)):
			cov_chol = SMM._cholesky(cv, min_covar)
			result[:, c] = SMM._mahalanobis_distance_chol(X, mu, cov_chol)
		return result

	@staticmethod
	def _mahalanobis_distance_mix_tied(X, means, covars, min_covar):
		cv = np.tile(covars, (means.shape[0], 1, 1))
		return SMM._mahalanobis_distance_mix_full(X, means, cv, min_covar)

	@staticmethod
	def _validate_covariances(covars, covariance_type, n_components):
		""" @brief Do basic checks on matrix covariance sizes and values"""
		
		if covariance_type == 'full':
			if len(covars.shape) != 3:
				raise ValueError("'full' covars must have shape (n_components, n_dim, n_dim)")
			elif covars.shape[1] != covars.shape[2]:
				raise ValueError("'full' covars must have shape (n_components, n_dim, n_dim)")
			for n, cv in enumerate(covars):
				if (not np.allclose(cv, cv.T) or np.any(linalg.eigvalsh(cv) <= 0)):
					raise ValueError("component %d of 'full' covars must be symmetric, positive-definite" % n)
				else:
					raise ValueError("covariance_type must be one of " + "'spherical', 'tied', 'diag', 'full'")
		
		elif covariance_type == 'diag':
			if len(covars.shape) != 2:
				raise ValueError("'diag' covars must have shape (n_components, n_dim)")
			elif np.any(covars <= 0):
				raise ValueError("'diag' covars must be non-negative")

		elif covariance_type == 'spherical':
			if len(covars) != n_components:
				raise ValueError("'spherical' covars have length n_components")
			elif np.any(covars <= 0):
				raise ValueError("'spherical' covars must be non-negative")

		elif covariance_type == 'tied':
			if covars.shape[0] != covars.shape[1]:
				raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
			elif (not np.allclose(covars, covars.T) or np.any(np.linalg.eigvalsh(covars) <= 0)):
				raise ValueError("'tied' covars must be symmetric, positive-definite")
	
	@staticmethod
	def distribute_covar_matrix_to_match_covariance_type(tied_cv, covariance_type, n_components):
		"""
		@brief Create all the covariance matrices from a given template.
		
		@param[in] tied_cv         Tied covariance that is going to be converted to other type.
		@param[in] covariance_type String that represents the type of the covariance parameters. 
											Must be one of 'spherical', 'tied', 'diag', 'full'.
		@param[in] n_components    Number of components in the mixture (integer value).
		
		@returns the tied covariance in the format specified by the user.
		"""

		if covariance_type == 'spherical':
			cv = np.tile(tied_cv.mean() * np.ones(tied_cv.shape[1]), (n_components, 1))
		elif covariance_type == 'tied':
			cv = tied_cv
		elif covariance_type == 'diag':
			cv = np.tile(np.diag(tied_cv), (n_components, 1))
		elif covariance_type == 'full':
			cv = np.tile(tied_cv, (n_components, 1, 1))
		else:
			raise ValueError("covariance_type must be one of " + "'spherical', 'tied', 'diag', 'full'")
		return cv
	
	@staticmethod
	def multivariate_t_rvs(m, S, df=np.inf, n=1):
		"""
		@brief Generate multivariate random variable sample from a t-Student distribution.
		@author Enzo Michelangeli

		Parameters
		----------
		m : array_like, mean of random variable, length determines dimension of random variable.

		S : array_like, square array of covariance  matrix.
	
		df : int or float, degrees of freedom.
	
		n : int, number of observations, return random array will be (n, len(m)).

		Returns
		-------
		rvs : ndarray, (n, len(m)), each row is an independent draw of a multivariate t distributed random 
		variable
		"""

		m = np.asarray(m)
		d = len(m)
		x = 1. if df == np.inf else np.random.chisquare(df, n) / df
		z = np.random.multivariate_normal(np.zeros(d), S, (n,))
	
		return m + z/np.sqrt(x)[:, None]   # same output format as random.multivariate_normal
	
	#@staticmethod
	#def sample(n_samples=1, n_pool=100):
	#	"""
	#	@brief Generate a random sample from the distribution using rejection sampling.
	#
	#	@param[in] n_samples TODO
	#		
	#	@param[in] n_pool    TODO
	#
	#	"""
	#		
	# TODO
	
	@property
	def n_components(self):
		"""@returns the number of distributions in the mixture"""
		return self._n_components
	
	@n_components.setter
	def n_components(self, n_components):
		self._n_components = n_components 
	
	@property
	def weights(self):
		"""@returns the weights of each component in the mixture"""
		return self.weights_
	
	@property
	def means(self):
		"""@returns the means of each component in the mixture"""
		return self.means_

	@property
	def degrees(self):
		"""@returns the degrees of freedom of each component in the mixture"""
		return self.degrees_

	@property
	def covariances(self):
		"""
		@brief Covariance parameters for each mixture component.
		@details The shape depends on the type of covariance matrix:
		
					(n_classes,  n_features)               if 'diag',
					(n_classes,  n_features, n_features)   if 'full'
					(n_classes,  n_features)               if 'spherical',
					(n_features, n_features)               if 'tied',
		
		@returns the covariance matrices for all the classes. 
		"""

		if self._covariance_type == 'full':
			return self.covars_
		elif self._covariance_type == 'diag':
			return [np.diag(cov) for cov in self.covars_]
		elif self._covariance_type == 'tied':
			return [self.covars_] * self._n_components
		elif self._covariance_type == 'spherical':
			return [np.diag(cov) for cov in self.covars_]
	
	# @brief Set the values for the covariance matrices.
	# @param[in] covars New covariance matrices.
	# @covariances.setter
	# def covariances(self, covars, cov_type):
	#	covars = np.asarray(covars)
	#	_validate_covariances(covars, cov_type, self._n_components)
	#	self.covars_ = covars
	
	# Class constants
	_covar_mstep_funcs = {
		'spherical': _covar_mstep_spherical.__func__,
		'diag': _covar_mstep_diag.__func__,
		'tied': _covar_mstep_tied.__func__, 
		'full': _covar_mstep_full.__func__,
	}
	
	_mahalanobis_funcs = {
		'spherical': _mahalanobis_distance_mix_spherical.__func__,
		'diag': _mahalanobis_distance_mix_diag.__func__,
		'tied': _mahalanobis_distance_mix_tied.__func__,
		'full': _mahalanobis_distance_mix_full.__func__,
	}

	_EPS = np.finfo(np.float64).eps
