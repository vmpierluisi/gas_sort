import numpy as np
from scipy.linalg import cholesky, cho_factor, cho_solve, solve_triangular
from filters.base_filter import BaseFilter

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class GASFilter(BaseFilter):
    """
    Generalized Autoregressive Score (GAS) filter for bounding-box tracking.

    State vector (12-dim):
        x_t = [x, y, a, h, x', y', a', h', x'', y'', a'', h'']
        where (x, y) = bounding-box centre, a = w/h, h = height.
        Primes denote velocities, double-primes accelerations.

    Measurement vector (4-dim):
        z_t = [x, y, a, h]

    -----------------------------------------------------------------------
    Observation model  (linear, time-invariant)
    -----------------------------------------------------------------------
        z_t = H x_t + v_t,        v_t ~ N(0, R_t)

    H = [I_4 | 0_4 | 0_4]  (4x12)  — we only observe positions.

    -----------------------------------------------------------------------
    Transition model  (linear, time-VARYING via GAS)
    -----------------------------------------------------------------------
        x_t = F_t x_{t-1} + u_t,  u_t ~ N(0, Q_t)

    F_0 is the standard constant-acceleration matrix; thereafter F_t is
    updated by the GAS recursion below.

    -----------------------------------------------------------------------
    GAS(1,1) recursion for F_t  (inverse-Fisher scaling)
    -----------------------------------------------------------------------
        s_t   = I_t^{-1}  nabla_t        (scaled score)
        F_t   = omega + alpha * s_t + beta * F_{t-1}

    where
        nabla_t  = d log p(z_t | z_{1:t-1}) / d F_{t-1}
                   (score of the predictive log-likelihood w.r.t. F)
        I_t      = E[-d^2 log p / d F^2]  (Fisher information for F)
        omega    = (1 - beta) * F_0        (intercept, ensures F -> F_0
                                            when innovations vanish)
        alpha    = learning rate for the score signal
        beta     = exponential-smoothing weight on the previous F

    See the `update` method for the full derivation of nabla_t, I_t, and
    their ratio.
    -----------------------------------------------------------------------
    """

    def __init__(self, alpha=0.02, beta=0.9):
        """
        Parameters
        ----------
        alpha : float
            GAS learning rate — scales the score contribution to F_t.
        beta : float
            GAS smoothing weight — controls how much of F_{t-1} carries over.
        """
        # GAS hyper-parameters
        self.alpha = alpha
        self.beta = beta

        # Defining the Transition matrix
        dt = 1
        F0 = np.eye(12)
        F0[0:4, 4:8]   = dt * np.eye(4)
        F0[0:4, 8:12]  = ((dt ** 2) / 2) * np.eye(4)
        F0[4:8, 8:12]  = dt * np.eye(4)
        self._transition_matrix = F0

        # Defining the Measurement matrix
        self._measurement_matrix = np.eye(4, 12)
        self.fisher_diag = np.ones((12, 12)) * 1e-6
        self.fisher_count=0

        # GAS intercept omega
        # omega = (1 - beta) * F_0 so that F -> F_0 at steady state.
        self.omega = (1 - self.beta) * F0

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model.
        self._std_weight_position     = 1 / 20    # position noise
        self._std_weight_velocity     = 1 / 160   # velocity noise
        self._std_weight_acceleration = 1 / 1280  # acceleration noise
        self._std_weight_aspect_p     = 1     # aspect-ratio position noise
        self._std_weight_aspect_v     = 1      # aspect-ratio velocity noise
        self._std_weight_aspect_a     = 1      # aspect-ratio acceleration noise

    def _noise_matrices(self, mean):
        """
        Build Q (motion_noise) and R (obs_noise) scaled by the current track height.
        """
        h = mean[3]
        # Calculating the measurement noise matrix
        var_measure = np.array([(self._std_weight_position*h)**2,
                                (self._std_weight_position*h)**2,
                                 self._std_weight_aspect_p**2,
                                (self._std_weight_position*h)**2])
        # Calculating the transition noise matrix
        var_trans = np.array([(self._std_weight_position*h)**2,
                              (self._std_weight_position*h)**2,
                               self._std_weight_aspect_p**2,
                              (self._std_weight_position*h)**2,
                              (self._std_weight_velocity*h)**2,
                              (self._std_weight_velocity*h)**2,
                               self._std_weight_aspect_v**2,
                              (self._std_weight_velocity*h)**2,
                              (self._std_weight_acceleration*h)**2,
                              (self._std_weight_acceleration*h)**2,
                               self._std_weight_aspect_a**2,
                              (self._std_weight_acceleration*h)**2
                              ])
        Q = np.diag(var_trans)
        R = np.diag(var_measure)
        return Q, R

    def initiate(self, measurement):
        """
        Create a new track from an unassociated detection.

        Parameters
        ----------
        measurement : ndarray, shape (4,)
            Bounding box as (x, y, a, h).

        Returns
        -------
        mean : ndarray, shape (12,)
            Initial state.  Positions from measurement; velocities and
            accelerations set to zero.
        covariance : ndarray, shape (12, 12)
            Initial covariance.  Velocities/accelerations get 10x larger
            std-dev to reflect high uncertainty from a single frame.
        F0 : ndarray, shape (12, 12)
            Initial transition matrix (the constant-acceleration default).
        """
        h = measurement[3]

        mean = np.zeros(12)
        mean[:4] = measurement[:4]

        # Initial covariance: 2x std for positions, 10x std for vel/acc
        init_var_trans = np.array([
            (2  * self._std_weight_position     * h)**2,
            (2  * self._std_weight_position     * h)**2,
            (2  * self._std_weight_aspect_p        )**2,
            (2  * self._std_weight_position     * h)**2,
            (10 * self._std_weight_velocity     * h)**2,
            (10 * self._std_weight_velocity     * h)**2,
            (10 * self._std_weight_aspect_v        )**2,
            (10 * self._std_weight_velocity     * h)**2,
            (10 * self._std_weight_acceleration * h)**2,
            (10 * self._std_weight_acceleration * h)**2,
            (10 * self._std_weight_aspect_a        )**2,
            (10 * self._std_weight_acceleration * h)**2,
        ])
        covariance = np.diag(init_var_trans)

        return mean, covariance, self._transition_matrix.copy()

    def predict(self, mean, covariance, F):
        Q, _ = self._noise_matrices(mean)

        mean = F @ mean
        covariance = F @ covariance @ F.T + Q

        return mean, covariance, F

    def update(self, mean, covariance, measurement, F):

        _, R = self._noise_matrices(mean)
        H = self._measurement_matrix  # (4, 12)
        R = H @ covariance @ H.T + R  # (4, 4)
        R_inv = np.linalg.inv(R)
        Fisher_inv = covariance @ H.T @ R_inv #PHR^-1

        # Innovation
        innovation = measurement - H @ mean  # (4,)

        # GAS score
        norm = max(np.dot(mean, mean), 1e-6)
        score = np.outer(Fisher_inv@innovation, mean) / norm #
        # H.T @ R_inv @ innovation
        # GAS recursion: update F
        new_F = self.omega + self.alpha * score + self.beta * F

        new_mean = mean + Fisher_inv @ innovation
        new_covariance = (np.eye(12) - Fisher_inv @ H) @ covariance

        return new_mean, new_covariance, new_F

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """
        Squared Mahalanobis distance for gating track–detection associations.

        Thresholds from `chi2inv95` (4 DOF full, 2 DOF position-only).

        Parameters
        ----------
        mean : ndarray, shape (12,)      — predicted state
        covariance : ndarray, (12, 12)   — predicted covariance
        measurements : ndarray, (N, 4)   — candidate detections
        only_position : bool             — use only (x, y) if True

        Returns
        -------
        ndarray, shape (N,) — squared Mahalanobis distances
        """
        #_, R = self._noise_matrices(mean)
        H = self._measurement_matrix  # (4, 12)
        #R = H @ covariance @ H.T + R  # (4, 4)

        mean = H @ mean
        covariance = H @ covariance @ H.T

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        #covariance = self._make_positive_definite(covariance)

        cholesky_factor = cholesky(covariance, lower=True)
        distance = measurements - mean
        z = solve_triangular(
            cholesky_factor, distance.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
