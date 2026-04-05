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
    Standard linear Kalman filter with a constant acceleration (CA) motion model.

    State vector (12-dim):
        x = [x, y, a, h, x', y', a', h', x'', y'', a'', h'']
    where (x, y) is bounding box centre, a = w/h, h is height.

    Measurement vector (4-dim):
        z = [x, y, a, h]

    KF System:
        x_k = Fx_{k-1} + w_k, with w ~ N(0, Q)
        z_k = Hx_{k-1} + v_k, with v ~ N(0, R)

        Prediction:
        x^_k|k-1 = Fx^_{k-1}
        P_k|k-1 = F @ P_k-1 @ F.T + Q

        Update:
        K_k = P_k|k-1 @ H.T @ (H @ P_k|k-1 @ H.T + R)^-1
        x^_k = x^_k|k-1 + K_k (z_k - H @ x^_k|k-1)
        P_k = (I - K_k @ H) @ P_k|k-1
    """

    def __init__(self, alpha=0.001, beta=0.95):
        # Defining the Transition matrix
        dt=1
        _transition_matrix = np.eye(12)
        _transition_matrix[0:4, 4:8] = dt * np.eye(4)
        _transition_matrix[0:4, 8:12] = ((dt ** 2) / 2) * np.eye(4)
        _transition_matrix[4:8, 8:12] = dt * np.eye(4)
        self._transition_matrix = _transition_matrix

        # Defining the Measurement matrix
        self._measurement_matrix = np.eye(4, 12)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model.
        self._std_weight_position = 1 / 20
        self._std_weight_velocity = 1 / 160
        self._std_weight_acceleration = 1 / 1280
        self._std_weight_aspect_p = 1e-2
        self._std_weight_aspect_v = 1e-3
        self._std_weight_aspect_a = 1e-4

        self.alpha = alpha
        self.beta = beta

    def _symmetrize(self, matrix):
        return 0.5 * (matrix + matrix.T)

    def _make_positive_definite(self, matrix, initial_jitter=1e-6, max_attempts=6):
        matrix = self._symmetrize(matrix)
        jitter = initial_jitter

        for _ in range(max_attempts):
            try:
                cholesky(matrix, lower=True, check_finite=False)
                return matrix
            except np.linalg.LinAlgError:
                matrix = matrix + jitter * np.eye(matrix.shape[0])
                jitter *= 10

        # Final fallback: clip tiny/negative eigenvalues to keep the matrix usable.
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.clip(eigvals, initial_jitter, None)
        matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return self._symmetrize(matrix)

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
        Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h)

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (12 dimensional) and covariance matrix (12x12
            dimensional) of the new track. Unobserved velocities and accelerations
            are initialized to 0 mean. Velocities and accelerations have been multiplied
            by 10 to reflect higher initial uncertainty after seeing a single frame.

        """

        h = measurement[3]
        mean = np.zeros(12)
        mean[:4] = measurement[:4]
        init_var_trans = np.array([(2*self._std_weight_position*h)**2,
                                   (2*self._std_weight_position*h)**2,
                                   (2*self._std_weight_aspect_p)**2,
                                   (2*self._std_weight_position*h)**2,
                                   (10*self._std_weight_velocity*h)**2,
                                   (10*self._std_weight_velocity*h)**2,
                                   (10*self._std_weight_aspect_v)**2,
                                   (10*self._std_weight_velocity*h)**2,
                                   (10*self._std_weight_acceleration*h)**2,
                                   (10*self._std_weight_acceleration*h)**2,
                                   (10*self._std_weight_aspect_a)**2,
                                   (10*self._std_weight_acceleration*h)**2
                                 ])
        covariance = np.diag(init_var_trans)
        F0 = self._transition_matrix
        omega = (1 - self.beta) * F0
        self.F0 = F0
        self.omega = omega

        return mean, covariance, F0

    def predict(self, mean, covariance, F):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 12 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 12x12 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities and accelerations are initialized to 0 mean.

        """
        Q, _ = self._noise_matrices(mean)

        mean = F @ mean
        covariance = F @ covariance @ F.T + Q
        covariance = self._make_positive_definite(covariance)

        return mean, covariance, F

    def update(self, mean, covariance, measurement, F):
        """
        Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (12 dimensional).
        covariance : ndarray
            The state's covariance matrix (12x12 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        _, R = self._noise_matrices(mean)

        S = self._measurement_matrix @ covariance @ self._measurement_matrix.T + R
        S = self._make_positive_definite(S)

        cholesky_factor, lower = cho_factor(
            S, lower=True, check_finite=False)

        K = cho_solve((cholesky_factor, lower),
                      np.dot(covariance, self._measurement_matrix.T).T,check_finite=False).T
        innovation = measurement - self._measurement_matrix @ mean
        new_mean = mean + K @ (measurement - self._measurement_matrix @ mean)
        new_covariance = (np.eye(12) - K @ self._measurement_matrix) @ covariance
        new_covariance = self._make_positive_definite(new_covariance)

        s = self._measurement_matrix.T @ np.outer(innovation, mean)
        new_F = self.omega + self.alpha * s + self.beta * F

        """
        U, s, Vt = np.linalg.svd(new_F)
        s = np.clip(s, 0.5, 2.0)
        new_F = U @ np.diag(s) @ Vt
        """

        rho = np.max(np.abs(np.linalg.eigvals(new_F)))
        if rho > 2.0:
            new_F = new_F * (2.0 / rho)


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
            Mean vector over the state distribution (12 dimensional).
        covariance : ndarray
            Covariance of the state distribution (12x12 dimensional).
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

        mean = self._measurement_matrix @ mean
        covariance = self._measurement_matrix @ covariance @ self._measurement_matrix.T + R


        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        covariance = self._make_positive_definite(covariance)

        cholesky_factor = cholesky(covariance, lower=True)
        distance = measurements - mean
        z = solve_triangular(
            cholesky_factor, distance.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
