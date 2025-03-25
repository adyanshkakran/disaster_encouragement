from numpy import array as vector
import numpy as np

class SEIIIRD_Model:

    def __init__(self, packed_parameters, time_dependent_params=None):
        print("Instantiate the SEIIIRD model ...")
        self.K = packed_parameters[0]
        self.N = packed_parameters[1]
        self.N_total = sum(self.N)
        self.beta_asym = packed_parameters[2]
        self.beta_sym = packed_parameters[3]
        self.beta_sev = packed_parameters[4]
        self.epsilon = packed_parameters[5]
        self.eta = packed_parameters[6]
        self.nu = packed_parameters[7]
        self.sigma = packed_parameters[8]
        self.gamma_asym = packed_parameters[9]
        self.gamma_sym = packed_parameters[10]
        self.gamma_sev_d = packed_parameters[11]
        self.gamma_sev_r = packed_parameters[12]
        self.beds = packed_parameters[13]
        
        self.time_dependent_params = time_dependent_params or {}

        self.numerical_tolerance_fine = 1e-6 # todo
        self.numerical_tolerance_coarse = 1e-3

    def eval_rhs(self, x, t):
        S = x[0 : self.K]
        E = x[self.K : 2*self.K]
        I_asym = x[2*self.K : 3*self.K]
        I_sym = x[3*self.K : 4*self.K]
        I_sev = x[4*self.K : 5*self.K]
        R = x[5*self.K : 6*self.K]
        D = x[6*self.K : 7*self.K]
        
        beta_asym = self.time_dependent_params.get('beta_asym', lambda t: self.beta_asym)(t)
        beta_sym = self.time_dependent_params.get('beta_sym', lambda t: self.beta_sym)(t)
        beta_sev = self.time_dependent_params.get('beta_sev', lambda t: self.beta_sev)(t)
        epsilon = self.time_dependent_params.get('epsilon', lambda t: self.epsilon)(t)
        eta = self.time_dependent_params.get('eta', lambda t: self.eta)(t)
        nu = self.time_dependent_params.get('nu', lambda t: self.nu)(t)
        sigma = self.time_dependent_params.get('sigma', lambda t: self.sigma)(t)
        gamma_asym = self.time_dependent_params.get('gamma_asym', lambda t: self.gamma_asym)(t)
        gamma_sym = self.time_dependent_params.get('gamma_sym', lambda t: self.gamma_sym)(t)
        gamma_sev_d = self.time_dependent_params.get('gamma_sev_d', lambda t: self.gamma_sev_d)(t)
        gamma_sev_r = self.time_dependent_params.get('gamma_sev_r', lambda t: self.gamma_sev_r)(t)
        
        # Sanity checks
        assert(x.shape[0] == 7 * self.K)
        # print(abs(sum(x) - sum(self.N)))
        assert(abs(sum(x) - sum(self.N)) < self.numerical_tolerance_coarse)
        assert(len(S) == self.K)
        assert(len(E) == self.K)
        assert(len(I_asym) == self.K)
        assert(len(I_sym) == self.K)
        assert(len(I_sev) == self.K)
        assert(len(R) == self.K)
        assert(len(D) == self.K)

        f_vec = np.empty(7*self.K)
        idx = 0
        overall_rhs = 0

        beta_factor = 1 # todo can be deleted after testing/debugging

        # ODE rhs for S
        for k in range(self.K):
            factor = 0.0
            for l in range(self.K):
                factor += beta_factor * beta_asym[l,k] * I_asym[l] + beta_factor * beta_sym[l,k] * I_sym[l] + beta_factor * beta_sev[l,k] * I_sev[l]
            assert(factor >= 0.0)
            f_vec[idx] = -factor * S[k] / self.N_total
            overall_rhs += f_vec[idx]
            idx += 1

        # ODE rhs for E
        for k in range(self.K):
            factor = 0.0
            for l in range(self.K):
                factor += beta_factor * beta_asym[l,k] * I_asym[l] + beta_factor * beta_sym[l,k] * I_sym[l] + beta_factor * beta_sev[l,k] * I_sev[l]
            assert(factor >= 0.0)
            f_vec[idx] = factor * S[k] / self.N_total - epsilon[k] * E[k]
            overall_rhs += f_vec[idx]
            idx += 1

        # ODE rhs for I_asym
        for k in range (self.K):
            f_vec[idx] = eta[k] * epsilon[k] * E[k] - gamma_asym * I_asym[k]
            overall_rhs += f_vec[idx]
            idx += 1

        # ODE rhs for I_sym
        for k in range(self.K):
            f_vec[idx] = (1 - eta[k]) * (1 - nu[k]) * epsilon[k] * E[k] - gamma_sym[k] * I_sym[k]
            overall_rhs += f_vec[idx]
            idx += 1

        # ODE rhs for I_sev
        for k in range(self.K):
            sigma_k = self._get_sigma_k(I_sev[k], k, sigma)
            f_vec[idx] = (1 - eta[k]) * nu[k] * epsilon[k] * E[k] - ((1.0 - sigma_k) * gamma_sev_r[k] + sigma_k * gamma_sev_d[k]) * I_sev[k]
            overall_rhs += f_vec[idx]
            idx += 1

        # ODE rhs for R
        for k in range(self.K):
            sigma_k = self._get_sigma_k(I_sev[k], k, sigma)
            f_vec[idx] = gamma_asym * I_asym[k] + gamma_sym[k] * I_sym[k] + (1.0 - sigma_k) * gamma_sev_r[k] * I_sev[k]
            overall_rhs += f_vec[idx]
            idx += 1

        # ODE rhs for D
        for k in range(self.K):
            sigma_k = self._get_sigma_k(I_sev[k], k, sigma)
            f_vec[idx] = sigma_k * gamma_sev_d[k] * I_sev[k]
            overall_rhs += f_vec[idx]
            idx += 1

        # Sanity checks
        assert(idx == 7 * self.K)
        assert(abs(overall_rhs) < self.numerical_tolerance_fine)

        return f_vec

    def _get_sigma_k(self, I_sev_k, k, sigma):
        assert(len(self.N) == self.K)
        beds_for_group_k = (self.N[k] / self.N_total) * self.beds
        if I_sev_k <= beds_for_group_k:
            return sigma[k]
        else:
            sigma_k = (sigma[k] * beds_for_group_k + I_sev_k - beds_for_group_k) / I_sev_k
            assert(sigma_k >= 0.0)
            assert(sigma_k <= 1.0)
            return sigma_k

# class SEIIIRD_Model
