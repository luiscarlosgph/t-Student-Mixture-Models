#!/usr/bin/env python
#
# @brief  Unit testing to check that the EM optimization for a mixture of t-Student distributions is working
#         as expected.
#
# @author Luis C. Garcia-Peraza Herrera (luis.herrera.14@ucl.ac.uk).
# @date   10 Jul 2017.

import unittest
import numpy as np
import scipy
import sklearn
import sklearn.utils
import sklearn.utils.estimator_checks
import os
import sys

# My imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/smm')))
import smm

# np.seterr(all='raise')

##
# @brief Prints the estimated parameters of a mixture of t-Students.
#
# @param[in] mix_t Object of the class SMM.


def print_params(name, mix_t, n_components, means, covars, dfs, weights):
    print('')
    print('==================================================')
    print(name)
    print('==================================================')
    print('t-Student mixture optimized parameters (correct values on the right):')
    print('')
    print('   Estimated component[s]: ' + str(mix_t.n_components))
    print('   Correct component[s]: ' + str(n_components))
    print('')
    print('   Estimated mean[s]:\n' + str(mix_t.means))
    print('   Correct mean[s]:\n' + str(means))
    print('')
    print('   Estimated covariance[s]:\n' + str(mix_t.covariances))
    print('   Correct covariance[s]:\n' + str(covars))
    print('')
    print('   Estimated degree[s] of freedom: ' + str(mix_t.degrees))
    print('   Correct degree[s] of freedom: ' + str(dfs))
    print('')
    print('   Estimated weight[s]: ' + str(mix_t.weights))
    print('   Correct weight[s]: ' + str(weights))
    print('')
    print('==================================================')


class TestSMM(unittest.TestCase):

    #def test_sklearn_check_estimator(self):
    #    sklearn.utils.estimator_checks.check_estimator(smm.SMM)

    def test_cholesky_cov_det(self):
        cv = np.array([[89, 3, 16], [3, 2, 1], [16, 1, 3]])
        min_covar = 1e-7
        eps = 1e-6
        correct_det = 2.0

        cov_chol = smm.SMM._cholesky(cv, min_covar)
        cov_det = np.power(np.prod(np.diagonal(cov_chol)), 2)
        self.assertTrue(np.fabs(correct_det - cov_det) < eps)

    def test_univariate_standard_t_student_density_full(self):
        eps = 1e-6
        means = np.array([0])
        covars = np.array([[[1]]])
        min_covar = 1e-6

        # Single-variable probability estimation test
        correct_proba = 0.06313533694826684
        X = np.array([[2]])
        dfs = np.array([7])
        proba = smm.SMM._multivariate_t_student_density_full(
            X, means, covars, dfs, min_covar)[0][0]
        self.assertTrue(np.fabs(correct_proba - proba) < eps)

        # Single-variable probability estimation test
        correct_proba = 0.1288320155351443
        X = np.array([[1.5]])
        dfs = np.array([25])
        proba = smm.SMM._multivariate_t_student_density_full(
            X, means, covars, dfs, min_covar)[0][0]
        self.assertTrue(np.fabs(correct_proba - proba) < eps)

        # Single-variable probability estimation test
        correct_proba = 0.014979288677507493
        X = np.array([[4.5]])
        dfs = np.array([1])
        proba = smm.SMM._multivariate_t_student_density_full(
            X, means, covars, dfs, min_covar)[0][0]
        self.assertTrue(np.fabs(correct_proba - proba) < eps)

    def test_univariate_t_student_density_full(self):
        eps = 1e-6
        means = np.array([31])
        covars = np.array([[[89]]])
        dfs = np.array([3])
        min_covar = 1e-6

        # Single-variable probability estimation test
        correct_proba = 0.022934460430541256823
        X = np.array([[40]])
        proba = smm.SMM._multivariate_t_student_density_full(
            X, means, covars, dfs, min_covar)[0][0]
        self.assertTrue(np.fabs(correct_proba - proba) < eps)

    def test_multivariate_t_student_density_full(self):
        eps = 1e-6
        means = np.array([[77, 3, 124]])
        covars = np.array([[[89, 3, 16], [3, 2, 1], [16, 1, 3]]])
        dfs = np.array([7])
        min_covar = 1e-6

        # Multi-variable probability estimation test
        correct_proba = 0.04951629782361236
        X = np.array([[77, 3, 124]])
        proba = smm.SMM._multivariate_t_student_density_full(
            X, means, covars, dfs, min_covar)[0][0]
        self.assertTrue(np.fabs(correct_proba - proba) < eps)

        # Multi-variable probability estimation test
        correct_scaled_proba = 3.028380928358
        X = np.array([[21.9, 1.2, 120.1]])
        proba = smm.SMM._multivariate_t_student_density_full(
            X, means, covars, dfs, min_covar)[0][0]
        scaled_proba = proba * 10**15
        self.assertTrue(np.fabs(correct_scaled_proba - scaled_proba) < eps)

    def test_multivariate_t_student_density_diag(self):
        eps = 1e-5
        means = np.array([[77, 3, 124]])
        covars = np.array([[89, 2, 3]])
        dfs = np.array([7])
        min_covar = 1e-6

        # Multi-variable probability estimation test
        correct_scaled_proba = 2.040467514702753
        X = np.array([[21.9, 1.2, 120.1]])
        proba = smm.SMM._multivariate_t_student_density_diag(
            X, means, covars, dfs, min_covar)[0][0]
        scaled_proba = proba * 10**7
        self.assertTrue(np.fabs(correct_scaled_proba - scaled_proba) < eps)

    def test_multivariate_t_student_density_spherical(self):
        eps = 1e-5
        means = np.array([[77, 3, 124]])
        covars = np.array([[89, 89, 89]])
        dfs = np.array([7])
        min_covar = 1e-6

        # Multi-variable probability estimation test
        correct_scaled_proba = 1.1638
        X = np.array([[21.9, 1.2, 120.1]])
        proba = smm.SMM._multivariate_t_student_density_spherical(
            X, means, covars, dfs, min_covar)[0][0]
        scaled_proba = proba * 10**8
        self.assertTrue(np.fabs(correct_scaled_proba - scaled_proba) < eps)

    def test_multivariate_t_student_density_tied(self):
        eps = 1e-6
        means = np.array([[77, 3, 124]])
        covars = np.array([[89, 3, 16], [3, 2, 1], [16, 1, 3]])
        dfs = np.array([7])
        min_covar = 1e-6

        # Multi-variable probability estimation test
        correct_proba = 0.04951629782361236
        X = np.array([[77, 3, 124]])
        proba = smm.SMM._multivariate_t_student_density_tied(
            X, means, covars, dfs, min_covar)[0][0]
        self.assertTrue(np.fabs(correct_proba - proba) < eps)

    def test_one_component_em(self):
        # Data
        n_samples = 500000
        n_dim = 3
        seed = 6
        np.random.seed(seed)

        # Epsilon thresholds for the error of the different parameters
        means_eps = n_dim * 1e-2
        covars_eps = (n_dim * (n_dim + 1) / 2.0) * 1e-1
        dfs_eps = 1e-1
        weights_eps = 1e-2

        # Parameters of the t-Student
        n_components = 1
        means = np.array([[77, 3, 124]])
        covars = np.array([[[89, 3, 16], [3, 2, 1], [16, 1, 3]]])
        weights = np.array([1.0])
        min_df = 1
        max_df = 10

        for i in range(min_df, max_df):
            dfs = np.array([i])

            # Generate random observations according to the parameters above
            # np.random.seed()
            obs = smm.SMM.multivariate_t_rvs(
                means[0], covars[0], df=dfs[0], n=n_samples)
            # obs = multivariate_t_rvs_rs(means, covars, dfs, n = n_samples, c = rejection_sampling_constant)

            # Fit mixture of t-students to data
            mix_t = smm.SMM(n_components=n_components, covariance_type='full', random_state=seed, tol=1e-12,
                            min_covar=1e-6, n_iter=500, n_init=1, params='wmcd', init_params='wmcd')
            mix_t.fit(obs)

            # Calculate estimation error
            mean_error = np.fabs(np.linalg.norm(means[0] - mix_t.means[0]))
            cov_error = np.fabs(np.linalg.norm(
                covars[0] - mix_t.covariances[0]))
            df_error = np.fabs(dfs[0] - mix_t.degrees[0])

            # Test that the optimized values are close to the real ones
            self.assertTrue(mean_error < means_eps)
            self.assertTrue(cov_error < covars_eps)
            self.assertTrue(df_error < dfs_eps)

            # Print results to screen, for debug purposes only
            # print('Mean error:', mean_error)
            # print('Covariance error:', cov_error)
            # print('Deg. of freedom error:', df_error)
            # print_params('test_1_component_em', mix_t, n_components, means, covars, dfs, weights)

    def test_several_component_em(self):
        n_dim = 3
        min_df = 1
        max_df = 10
        min_components = 2
        max_components = 5
        n_samples = 250000
        seed = 9
        np.random.seed(seed)

        for k in range(min_components, max_components):

            # Epsilon thresholds for the error of the different parameters
            means_eps = n_dim * 1e-1 * k
            covars_eps = (n_dim * (n_dim + 1) / 2.0) * float(k)
            dfs_eps = 1.0 * k
            weights_eps = 1e-1
            aic_eps = 0.01

            # Generate k random means
            means = np.random.randint(100, size=(k, n_dim))
            # print('Real means:', means)

            # Generate k random covariance matrices
            vector_covars = np.random.randint(
                100, size=(k, int((n_dim + 1) * n_dim / 2)))
            covars = np.zeros((k, n_dim, n_dim))
            for c in range(k):
                covars[c][np.triu_indices(n_dim)] = vector_covars[c]
                covars[c] = covars[c].T + covars[c] + 1.0
                # print('det[c]:', np.linalg.det(covars[c]))
                min_eig = np.min(np.real(np.linalg.eigvals(covars[c])))
                if min_eig < 0:
                    covars[c] -= 10 * min_eig * np.eye(*covars[c].shape)
            # print('Real covars:', covars)

            # Generate k random degrees of freedom
            dfs = np.random.randint(max_df, size=(k,)) + 1.0
            # print('Real dfs:', dfs)

            # Generate k weights, ensure that each weight is at least greater than a threshold (e.g. 0.1)
            weights = None
            thresh = 0.1
            weights_ok = False
            while not weights_ok:
                weights = np.random.rand(k)
                weights /= weights.sum()
                weights_ok = True
                for c in range(k):
                    if weights[c] < thresh:
                        weights_ok = False
            # print('Real weights:', weights)

            # Generate random observations according to the parameters above
            obs = []
            # obs = np.empty((k, n_samples, n_dim))
            for c in range(k):
                c_obs = smm.SMM.multivariate_t_rvs(
                    means[c], covars[c], df=dfs[c], n=n_samples)
                c_obs = c_obs[0:int(np.round(n_samples * weights[c]))]
                obs.append(c_obs)
            obs = np.vstack(obs)
            # print('Real obs:', obs)

            # Fit mixture of t-students to data
            mix_t = smm.SMM(n_components=k, covariance_type='full', random_state=seed, tol=1e-12,
                            min_covar=1e-6, n_iter=10000, n_init=1, params='wmcd', init_params='wmcd')
            mix_t.fit(obs)

            # Match the extracted distributions to the original ones
            indices = [x for x in range(k)]
            for c in range(k):
                min_match_val = 1000
                min_match_ind = 0
                for c_res in range(k):
                    # If we have a match... (assuming the weights are properly computed)
                    weight_diff = np.fabs(weights[c] - mix_t.weights[c_res])
                    if weight_diff < min_match_val:
                        min_match_val = weight_diff
                        min_match_ind = c_res
                indices[c] = min_match_ind

            # Check that the error in the estimation of each parameter is under the threshold
            for c in range(k):
                # Check that the means match
                mean_error = np.linalg.norm(means[c] - mix_t.means[indices[c]])
                # print('Correct mean: ' + str(means[c]))
                # print('Estimated mean: ' + str(mix_t.means[indices[c]]))
                self.assertTrue(mean_error < means_eps)

                # Check that the covariances match
                cov_error = np.linalg.norm(
                    covars[c] - mix_t.covariances[indices[c]], 'fro')
                self.assertTrue(cov_error < covars_eps)

                # Check that the degrees of freedom match
                df_error = np.fabs(float(dfs[c]) - mix_t.degrees[indices[c]])
                self.assertTrue(df_error < dfs_eps)

            # Print results to screen, debug purposes only
            # print_params('test_' + str(k) + '_component_em', mix_t, k, means, covars, dfs, weights)

            # Check that the difference of AIC is small engough
            estimated_aic = mix_t.aic(obs)
            mix_t.means_ = means
            mix_t.covars_ = covars
            mix_t.degrees_ = dfs
            mix_t.weights_ = weights
            real_aic = mix_t.aic(obs)
            self.assertTrue(np.fabs(real_aic - estimated_aic) < aic_eps)

if __name__ == '__main__':
    unittest.main()
