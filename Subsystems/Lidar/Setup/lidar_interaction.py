from asyncio import sleep
import numpy as np
from matplotlib import pyplot as plt, animation
from rplidar import RPLidar
from os import path

BAUDRATE: int = 115200
DEVICE_PATH: str = 'COM4'
TIMEOUT: int = 1
IMIN: int = 0
IMAX: int = 50
DMAX: int = 4000

def read_info():
    if path.exists(DEVICE_PATH):
        print('\n{:*^50s}'.format(" Lidar Status "))
        print('Found device : {0}'.format(DEVICE_PATH))

        lidar = RPLidar(port=DEVICE_PATH, timeout=TIMEOUT, baudrate=BAUDRATE)
        lidar_info = lidar.get_info()

        for key, value in lidar_info.items():
            print('{0:<13}: {1}'.format(key.capitalize(), str(value)))

        health = lidar.get_health()
        print('Health Status: {0[0]} - {0[1]}'.format(health))

        print('*' * 50)

        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

    else:
        print('[Error] Could not found device: {0}'.format(DEVICE_PATH))


def read_scan_point():
    def read_next_scan(iterator):
        next_scan = next(iterator)
        return next_scan

    if path.exists(DEVICE_PATH):
        print('\n{:*^50s}'.format(" Lidar Scans "))
        print('Found device : {0}'.format(DEVICE_PATH))

        lidar = RPLidar(port=DEVICE_PATH, timeout=TIMEOUT, baudrate=BAUDRATE)
        scans_iterator = lidar.iter_scans(scan_type='express')


        while True:
            scan = read_next_scan(scans_iterator)
            point = scan[0]
            print(f"Scan point: {point}")
    else:
        print('[Error] Could not found device: {0}'.format(DEVICE_PATH))


def plot_scans():
    def update_line(num, iterator, line):
        scan = next(iterator)
        offsets = np.array([(np.radians(meas[1]), meas[2]) for meas in scan])
        line.set_offsets(offsets)
        return line

    if path.exists(DEVICE_PATH):
        print('\n{:*^50s}'.format(" Lidar Scans "))
        print('Found device : {0}'.format(DEVICE_PATH))

        lidar = RPLidar(port=DEVICE_PATH, timeout=TIMEOUT, baudrate=BAUDRATE)

        fig = plt.figure()
        title = 'RPLIDAR'
        fig.set_label(title)
        fig.canvas.manager.set_window_title(title)

        ax = plt.subplot(111, projection='polar')
        line = ax.scatter([0, 0], [0, 0], s=5, c=[IMIN, IMAX], cmap=plt.cm.Greys_r, lw=0)
        ax.set_title('360° scan result')
        ax.set_rmax(DMAX)
        ax.grid(True)

        scans_iterator = lidar.iter_scans(scan_type='express')
        ani = animation.FuncAnimation(fig, update_line, fargs=(scans_iterator, line), interval=10, cache_frame_data=False)

        plt.show()
        lidar.stop()
        lidar.disconnect()

    else:
        print('[Error] Could not found device: {0}'.format(DEVICE_PATH))

if __name__ == '__main__':
    # read_info()
    # read_scan_point()
    plot_scans()