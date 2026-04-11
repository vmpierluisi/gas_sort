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


class KalmanFilter(BaseFilter):
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

    def __init__(self):
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
        return mean, covariance, None

    def predict(self, mean, covariance, F=None):
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

        mean = self._transition_matrix @ mean
        covariance = self._transition_matrix @ covariance @ self._transition_matrix.T + Q

        return mean, covariance, None

    def update(self, mean, covariance, measurement, F=None):
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

        cholesky_factor, lower = cho_factor(
            S, lower=True, check_finite=False)

        K = cho_solve((cholesky_factor, lower),
                      np.dot(covariance, self._measurement_matrix.T).T,check_finite=False).T

        new_mean = mean + K @ (measurement - self._measurement_matrix @ mean)
        new_covariance = (np.eye(12) - K @ self._measurement_matrix) @ covariance

        return new_mean, new_covariance, None

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

        cholesky_factor = cholesky(covariance, lower=True)
        distance = measurements - mean
        z = solve_triangular(
            cholesky_factor, distance.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
