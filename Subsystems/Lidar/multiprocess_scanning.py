import multiprocessing
from os import path
from rplidar import RPLidar
import matplotlib.pyplot as plt

from feature_extraction import get_corners
from scan_formatting import *

# Scanner Settings
SCANNER_BAUDRATE: int = 115200
SCANNER_DEVICE_PATH: str = 'COM4'
SCANNER_TIMEOUT: int = 1 # In seconds

# Scan Parameters
SCAN_MAX_DISTANCE: float = 5 # Farthest point from the scanner to consider

# Feature Extraction Parameters
FEATURE_ANGLE_THRESHOLD: float = 45 # Angle at which to start considering corners
FEATURE_DISTANCE_THRESHOLD: float = 0.1 # Distance at which to cluster and average corners

def get_scans(scansQueue: multiprocessing.Queue, stopQueue: multiprocessing.Queue):
    lidar = RPLidar(port=SCANNER_DEVICE_PATH, timeout=SCANNER_TIMEOUT, baudrate=SCANNER_BAUDRATE)
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


def scans_processing(scansQueue: multiprocessing.Queue, stopQueue: multiprocessing.Queue, mustPlot=False):
    scan_counter=0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    while scan_counter<100:
        current_scan = scansQueue.get()
        if current_scan is not None:
            scan_counter+=1
            polar_scan = format_scan(scan=current_scan, distance_threshold=SCAN_MAX_DISTANCE)
            polar_corners = get_corners(polar_scan, distance_threshold=FEATURE_DISTANCE_THRESHOLD, angle_threshold=np.deg2rad(FEATURE_ANGLE_THRESHOLD))

            if mustPlot:
                ax.clear()
                ax.plot(polar_scan[:, 1], polar_scan[:, 0], 'go', markersize=2, label="Points")
                if polar_corners.size > 0:
                    ax.plot(polar_corners[:, 1], polar_corners[:, 0], 'r*', markersize=5, label="Corners")
                ax.set_rmax(SCAN_MAX_DISTANCE)
                ax.set_title(f"Scan {scan_counter}")
                ax.legend(loc="upper right")
                plt.pause(0.01)

    stopQueue.put("Stop")
    plt.ioff()
    plt.close()


def main():
    if not path.exists(SCANNER_DEVICE_PATH):
        print("Ensure Lidar scanner is plugged in.")
        return
    else:
        scans_queue = multiprocessing.Queue(maxsize=1)
        stop_queue = multiprocessing.Queue(maxsize=1)

        get_scans_process = multiprocessing.Process(target=get_scans, args=(scans_queue, stop_queue))
        process_scans_process = multiprocessing.Process(target=scans_processing, args=(scans_queue, stop_queue, True))

        get_scans_process.start()
        process_scans_process.start()

        print("Process information:")
        print(f"Get Scans PID: {get_scans_process.pid}")
        print(f"Process Scans PID: {process_scans_process.pid}")

        get_scans_process.join()
        process_scans_process.join()


if __name__=="__main__":
    main()
