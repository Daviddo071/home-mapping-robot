import numpy as np

def pi2pi(angle: float) -> float:
    """
    Wraps an angle provided in radians to be between -pi and pi
    :param angle: Input angle in radians
    :return:
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def format_scan(scan: list, distance_threshold: float=6):
    """
    Takes in a list of tuples with (x, theta [degrees], distance [m])
    Returns a numpy array of [r [m], theta [rad]]
    :param distance_threshold:
    :param scan:
    :return:
    """
    result = np.array([[scan_point[2]/1000, pi2pi(-np.deg2rad(scan_point[1]))] for scan_point in scan])
    result = result[result[:, 0] < distance_threshold]
    result = result[result[:, 1].argsort()]
    return result
