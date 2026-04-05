from filters.kalman_filter import KalmanFilter

class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter with CA motion model.

    Under a linear Gaussian CA model with linear observations,
    the Jacobians J_F = F and J_H = H exactly, so EKF reduces
    to the standard KF numerically.
    """
    pass