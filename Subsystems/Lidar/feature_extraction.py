import numpy as np
from sklearn.cluster import DBSCAN


def cluster_and_average(points_with_duplicates: np.ndarray, neighbourhood_radius: float, min_samples: int=1) -> np.ndarray:
    """
    Uses sklearn.DBSCAN to cluster an input that has duplicates.
    Averages points within the cluster to get a feature location.

    :param points_with_duplicates: Input points. Shape of (n, 2)
    :param neighbourhood_radius: Max distance with which points are considered.
                                 Smaller value means possible duplicates, larger value = poor clustering.
    :param min_samples: Minimum number of points in a cluster. Default is 1.
    :return:
    """
    points_with_duplicates = np.array(points_with_duplicates)
    if points_with_duplicates.shape[0]==0:
        return points_with_duplicates

    sk_clustering = DBSCAN(eps=neighbourhood_radius, min_samples=min_samples).fit(points_with_duplicates)
    cluster_labels = sk_clustering.labels_

    averaged_points = []
    for label in set(cluster_labels):
        cluster_points = points_with_duplicates[cluster_labels == label]
        mean_point = cluster_points.mean(axis=0)
        averaged_points.append(mean_point)

    return np.array(averaged_points)


def get_corners(polar_scan: np.ndarray, angle_threshold: float = np.deg2rad(45), distance_threshold: float = 0.5,
                window_size: int = 15, neighbourhood_radius: float = 0.075) -> np.ndarray:
    """
    Features to extract are corners.

    :param polar_scan: Polar scan, nx2 (r, theta)
    :param angle_threshold: Threshold used to determine corners in the scan
    :param distance_threshold: Distance threshold of windows to consider when finding corners.
    :param window_size: Amount of points to consider at a time.
    :param neighbourhood_radius: Closeness of points for them to be considered as residing in the same neighborhood
    :return:
    """
    window_size = window_size if window_size%2==1 else window_size+1
    num_tests = window_size//2
    num_positives_needed = num_tests//2 + 1

    points_cartesian = np.array([polar_scan[:,0] * np.cos(polar_scan[:,1]), polar_scan[:,0] * np.sin(polar_scan[:,1])]).T
    windowed_corners = []

    for k in range(0, len(points_cartesian)-window_size):
        window = points_cartesian[k:k+window_size]
        centre_point = window[window_size//2]
        distances = np.linalg.norm(window - centre_point, axis=1)
        if max(distances) > distance_threshold:
            continue
        else:
            window_1 = window[0:window_size//2]
            centre_point = np.array(window[window_size//2])
            window_2 = window[window_size//2+1:]

            rise1s = window_1[:,1] - centre_point[1]
            run1s = window_1[:,0] - centre_point[0]
            run1s_safe = np.where(run1s == 0, 1e-5, run1s)
            m1 = rise1s/run1s_safe

            rise2s = window_2[:,1] - centre_point[1]
            run2s = window_2[:,0] - centre_point[0]
            run2s_safe = np.where(run2s == 0, 1e-5, run2s)
            m2 = rise2s/run2s_safe

            gradient_products = m1 * m2
            gradient_products_safe = np.where(gradient_products == 1, 1+1e-5, gradient_products)

            angles = np.arctan(np.abs((m1-m2)/(1+gradient_products_safe)))
            angles_thresholded = np.where(angles > angle_threshold, 1, 0)
            if np.sum(angles_thresholded)>num_positives_needed:
                windowed_corners.append(centre_point)

    corners = np.array(windowed_corners)
    corners = np.reshape(corners, (corners.shape[0], 2))
    features = cluster_and_average(corners, neighbourhood_radius)

    features_polar = np.array([
        np.linalg.norm(features, axis=1),
        np.arctan2(features[:,1], features[:,0]),
    ]).T

    return features_polar
