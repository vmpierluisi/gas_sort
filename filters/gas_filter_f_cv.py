import numpy as np
from scipy.linalg import cholesky, solve_triangular
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
     GAS(1, 1)  filter with a constant velocity (CV) motion model and KF update step.

    State vector (12-dim):
        q = [x, y, a, h, x', y', a', h']
    where (x, y) is bounding box centre, a = w/h, h is height.

    Measurement vector (4-dim):
        z = [x, y, a, h]

    GAS System:
        q_t = Fq_{t-1} + w_t, with w ~ N(0, Q)
        z_t = Hx_{t-1} + v_t, with v ~ N(0, R)

        Prediction:
        q_t|t-1 = Fq_{t-1}
        P_t|t-1 = F @ P_t-1 @ F.T + Q

        Update:
        K_t = P_t|t-1 @ H.T @ (H @ P_t|t-1 @ H.T + R)^-1
        q_t = q_t|t-1 + K_t (z_t - H @ q_t|t-1)

        P_t = F @ P_t|t-1 @ F.T + Q 
        or 
        P_t = (I - K_t @ H) @ P_t|t-1 @ (I - K_t @ H).T + K_t @ R @ K_t.T
    """

    def __init__(self, alpha=0.2, beta=0.8):
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
        F0 = np.eye(8)
        F0[0:4, 4:8] = dt * np.eye(4)
        self._transition_matrix = F0

        # Defining the Measurement matrix
        self._measurement_matrix = np.eye(4, 8)
        self.fisher_diag = np.ones((8, 8)) * 1e-6
        self.fisher_count=0

        # GAS intercept omega
        # omega = (1 - beta) * F_0 so that F -> F_0 at steady state.
        self.omega = (1 - self.beta) * F0

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model.
        self._std_weight_position     = 1 / 20    # position noise
        self._std_weight_velocity     = 1 / 160   # velocity noise
        self._std_weight_aspect_p     = 1e-2     # aspect-ratio position noise
        self._std_weight_aspect_v     = 1e-3      # aspect-ratio velocity noise

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
                              (self._std_weight_velocity*h)**2
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
        mean : ndarray, shape (8,)
            Initial state.  Positions from measurement; velocities set to zero.
        covariance : ndarray, shape (8, 8)
            Initial covariance.  Velocities get 10x larger std-dev to reflect
            high uncertainty from a single frame.
        F0 : ndarray, shape (8, 8)
            Initial transition matrix (the constant-velocity default).
        """
        h = measurement[3]

        mean = np.zeros(8)
        mean[:4] = measurement[:4]

        # Initial covariance: 2x std for positions, 10x std for vel
        init_var_trans = np.array([
            (2  * self._std_weight_position     * h)**2,
            (2  * self._std_weight_position     * h)**2,
            (2  * self._std_weight_aspect_p        )**2,
            (2  * self._std_weight_position     * h)**2,
            (10 * self._std_weight_velocity     * h)**2,
            (10 * self._std_weight_velocity     * h)**2,
            (10 * self._std_weight_aspect_v        )**2,
            (10 * self._std_weight_velocity     * h)**2,
        ])
        covariance = np.diag(init_var_trans)

        return mean, covariance, self._transition_matrix.copy()

    def predict(self, mean, covariance, F):
        """
        Run GAS(1, 1) filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        Q, R = self._noise_matrices(mean) # Remove R

        mean = F @ mean
        covariance = F @ covariance @ F.T + Q

        return mean, covariance, F

    def update(self, mean, covariance, measurement, F):
        """
        Run GAS(1, 1) filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        Q, R = self._noise_matrices(mean)
        H = self._measurement_matrix

        # Calculating P^-1
        R_pred = H @ covariance @ H.T + R
        R_inv = np.linalg.inv(R_pred)
        K = covariance @ H.T @ R_inv

        # Innovation
        innovation = measurement - H @ mean

        # GAS score
        lam1, lam2 = 1e-3, 1e-3
        scale_1 = H.T @ R_inv @ H + lam1 * np.eye(8)
        scale_2 = np.outer(mean, mean) + lam2 * np.eye(8)
        #norm = max(np.dot(mean, mean), 1e-6)
        score_raw = np.outer(H.T @ R_inv @ innovation, mean)
        score = np.linalg.solve(scale_1, score_raw) @ np.linalg.inv(scale_2)

        # Update F
        new_F = self.omega + self.alpha * score + self.beta * F

        # Update mean and covariance
        new_mean = mean + K @ innovation
        new_covariance = new_F @ covariance @ new_F.T + Q

        # Joseph form KF covariance update
        #IKH = np.eye(8) - K @ H
        #new_covariance = IKH @ covariance @ IKH.T + K @ R_pred @ K.T

        return new_mean, new_covariance, new_F

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """
        Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        _, R = self._noise_matrices(mean)
        H = self._measurement_matrix  # (4, 8)

        mean = H @ mean
        covariance = H @ covariance @ H.T + R

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = cholesky(covariance, lower=True)
        distance = measurements - mean
        z = solve_triangular(
            cholesky_factor, distance.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha