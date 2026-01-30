from matplotlib import pyplot as plt
from rplidar import RPLidar
from os import path

from feature_extraction import get_corners
from scan_formatting import *

BAUDRATE: int = 115200
DEVICE_PATH: str = 'COM4'
TIMEOUT: int = 1


def plot_scan():
    if path.exists(DEVICE_PATH):
        print('\n{:*^50s}'.format(" Lidar Scans "))
        print('Found device : {0}'.format(DEVICE_PATH))

        lidar = RPLidar(port=DEVICE_PATH, timeout=TIMEOUT, baudrate=BAUDRATE)
        scans_iterator = lidar.iter_scans(scan_type='express')
        scan = next(scans_iterator)
        lidar.stop()
        lidar.disconnect()

        polar_scan = format_scan(scan)
        polar_corners = get_corners(polar_scan, distance_threshold=0.5)

        figure = plt.figure()
        ax = figure.add_subplot(111, projection='polar')
        ax.plot(polar_scan[:, 1], polar_scan[:, 0], 'go', markersize=2, label='Scan points')
        ax.plot(polar_corners[:, 1], polar_corners[:, 0], 'r*', markersize=5, label='Corner points')
        ax.legend()
        plt.show()

    else:
        print('[Error] Could not find device: {0}'.format(DEVICE_PATH))


if __name__ == '__main__':
    for k in range(0, 1):
        plot_scan()
