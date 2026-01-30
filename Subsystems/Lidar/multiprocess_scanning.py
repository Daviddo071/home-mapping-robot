import multiprocessing
from os import path
from rplidar import RPLidar
import matplotlib.pyplot as plt

from feature_extraction import get_corners
from scan_formatting import *

# Scanner Settings
BAUDRATE: int = 115200
DEVICE_PATH: str = 'COM4'
TIMEOUT: int = 1

# Scan Parameters
MAX_DISTANCE: float = 5 # Farthest point from the scanner to consider

# Feature Extraction Parameters
ANGLE_THRESHOLD: float = 60 # Angle at which to start considering corners
DISTANCE_THRESHOLD: float = 0.1 # Distance at which to cluster and average corners

def get_scans(scansQueue: multiprocessing.Queue, stopQueue: multiprocessing.Queue):
    lidar = RPLidar(port=DEVICE_PATH, timeout=TIMEOUT, baudrate=BAUDRATE)
    lidar.start()
    lidar.stop()

    for scan in lidar.iter_scans():
        if not scansQueue.full():
            scansQueue.put(scan)
        else:
            scansQueue.get()
            scansQueue.put(scan)

        if stopQueue.full():
            lidar.stop()
            lidar.disconnect()
            return


def other_processing(scansQueue: multiprocessing.Queue, stopQueue: multiprocessing.Queue, mustPlot=False):
    scan_counter=0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    while scan_counter<100:
        current_scan = scansQueue.get()
        if current_scan is not None:
            scan_counter+=1
            polar_scan = format_scan(scan=current_scan, distance_threshold=MAX_DISTANCE)
            polar_corners = get_corners(polar_scan)

            if mustPlot:
                ax.clear()
                ax.plot(polar_scan[:, 1], polar_scan[:, 0], 'go', markersize=2, label="Scan Points")
                if polar_corners.size > 0:
                    ax.plot(polar_corners[:, 1], polar_corners[:, 0], 'r*', markersize=5, label="Corners")
                ax.set_rmax(MAX_DISTANCE)
                ax.set_title(f"Polar Scan {scan_counter}")
                ax.legend(loc="upper right")
                plt.pause(0.01)

    stopQueue.put("Stop")
    plt.ioff()
    plt.close()


def main():
    if not path.exists(DEVICE_PATH):
        print("Ensure Lidar scanner is plugged in.")
        return
    else:
        scans_queue = multiprocessing.Queue(maxsize=1)
        stop_queue = multiprocessing.Queue(maxsize=1)

        get_scans_process = multiprocessing.Process(target=get_scans, args=(scans_queue, stop_queue))
        process_scans_process = multiprocessing.Process(target=other_processing, args=(scans_queue, stop_queue, True))

        get_scans_process.start()
        process_scans_process.start()

        print("Process information:")
        print(f"Get Scans PID: {get_scans_process.pid}")
        print(f"Process Scans PID: {process_scans_process.pid}")

        get_scans_process.join()
        process_scans_process.join()


if __name__=="__main__":
    main()
