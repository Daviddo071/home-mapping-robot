import numpy as np


def wrap_angle(x, zero_2_2pi=False, degree=False):
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle

def format_scan(scan: list, distance_threshold: float=6):
    """
    Takes in a list of tuples with (x, theta [degrees], distance [m])
    Returns a numpy array of [r [m], theta [rad]]
    :param distance_threshold:
    :param scan:
    :return:
    """
    result = np.array([[scan_point[2]/1000, wrap_angle(-np.deg2rad(scan_point[1]))] for scan_point in scan])
    result = result[result[:, 0] < distance_threshold]
    result = result[result[:, 1].argsort()]
    return result
