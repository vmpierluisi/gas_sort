
from abc import ABC, abstractmethod
import numpy as np


class BaseFilter(ABC):
    """
    Abstract base class for all filter variants in DeepGAS.

    All filters operate on a 12-dimensional constant acceleration (CA) state:
        x = [x, y, a, h, x', y', a', h', x'', y'', a'', h'']
    where (x, y) is the bounding box centre, a = w/h is the aspect ratio,
    h is the height, primes denote velocities, and double-primes denote
    accelerations.

    Measurements are 4-dimensional:
        z = [x, y, a, h]
    """

    @abstractmethod
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a new track from an unassociated measurement.

        Parameters
        ----------
        measurement : np.ndarray, shape (4,)
            Observed bounding box [x, y, a, h].

        Returns
        -------
        mean : np.ndarray, shape (12,)
            Initial state mean. Velocities and accelerations initialised to 0.
        covariance : np.ndarray, shape (12, 12)
            Initial state covariance.
        """

    @abstractmethod
    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate state distribution one time step forward.

        Parameters
        ----------
        mean : np.ndarray, shape (12,)
        covariance : np.ndarray, shape (12, 12)

        Returns
        -------
        mean : np.ndarray, shape (12,)
        covariance : np.ndarray, shape (12, 12)
        """

    @abstractmethod
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Incorporate a new measurement into the state estimate.

        Parameters
        ----------
        mean : np.ndarray, shape (12,)
        covariance : np.ndarray, shape (12, 12)
        measurement : np.ndarray, shape (4,)

        Returns
        -------
        mean : np.ndarray, shape (12,)
        covariance : np.ndarray, shape (12, 12)
        """

    @abstractmethod
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """
        Compute Mahalanobis distance between state and a set of measurements.
        Used by the tracker to gate association candidates.

        Parameters
        ----------
        mean : np.ndarray, shape (12,)
        covariance : np.ndarray, shape (12, 12)
        measurements : np.ndarray, shape (N, 4)
        only_position : bool
            If True, use only (x, y) for the distance computation.

        Returns
        -------
        distances : np.ndarray, shape (N,)
        """