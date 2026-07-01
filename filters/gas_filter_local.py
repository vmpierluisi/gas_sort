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


class GASFilterPred(BaseFilter):
    """
    Gaussian location filter (no motion model).

    State vector (4-dim):
        q = [x, y, a, h]
    where (x, y) is bounding box centre, a = w/h, h is height.

    Measurement vector (4-dim):
        z = [x, y, a, h]

    Gaussian location filter:
        mu_t = alpha * (z_t - mu_t) + beta * mu_t-1
    """

    def __init__(self, alpha=1, beta=1):
        """
        Parameters
        ----------
        alpha : float
            GAS learning rate — scales the innovation (z_t - mu) into mu_t.
        beta : float
            GAS smoothing weight — controls how much of mu_t-1 carries over.
        """
        self._measurement_matrix = np.eye(4, 12)

        # GAS hyper-parameters
        self.alpha = alpha
        self.beta = beta

        # Observation uncertainty is chosen relative to the current state
        # estimate. This filter has no motion model, so only the measurement
        # (position/aspect) noise weights are needed.
        self._std_weight_position = 1 / 20    # position noise
        self._std_weight_aspect_p = 1e-2      # aspect-ratio position noise

    def _measurement_noise(self, mean):
        """
        Build R (obs_noise), a 4x4 diagonal scaled by the current track height.
        """
        h = mean[3]
        var_measure = np.array([(self._std_weight_position*h)**2,
                                (self._std_weight_position*h)**2,
                                 self._std_weight_aspect_p**2,
                                (self._std_weight_position*h)**2])
        return np.diag(var_measure)

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
            accelerations left at zero (unused by this location filter).
        covariance : ndarray, shape (4, 4)
            Initial position/aspect covariance.
        """
        h = measurement[3]

        mean = np.zeros(12)
        mean[:4] = measurement[:4]

        
        init_var = np.array([
            (2 * self._std_weight_position * h)**2,
            (2 * self._std_weight_position * h)**2,
            (2 * self._std_weight_aspect_p    )**2,
            (2 * self._std_weight_position * h)**2,
        ])
        covariance = np.diag(init_var)

        return mean, covariance, None

    def predict(self, mean, covariance, F=None):
        """
        Run GAS(1, 1) filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 12 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 4x4 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            The mean vector and covariance matrix, returned unchanged — this
            location filter has no motion model to propagate.

        """
        # No motion model: the location filter leaves the state unchanged and
        # relies entirely on the measurement update.
        return mean, covariance, None

    def update(self, mean, covariance, measurement, F=None):
        """
        Run GAS(1, 1) filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (12 dimensional).
        covariance : ndarray
            The state's covariance matrix (4x4 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """

        H = self._measurement_matrix

        # Innovation = score when we scale by sigma^2
        innovation = measurement - H @ mean # (4, )

        # Update mean, covariance
        new_mean = mean.copy()
        new_mean[:4] = self.alpha * innovation + self.beta * (H @ mean)

        new_covariance = self._measurement_noise(mean)

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
            Mean vector over the state distribution (12 dimensional; projected
            to 4 dimensions internally via the measurement matrix).
        covariance : ndarray
            Covariance of the state distribution (4x4 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h)
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

        H = self._measurement_matrix  # (4, 12)
        mean = H @ mean

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
