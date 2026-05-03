import matplotlib.pyplot as plt
import numpy as np

DT = 0.2 # Simulation time steps (s)
SIM_TIME = 40*np.pi # Total simulation time (s)
SCAN_MAX = 10 # Max effective distance (m) of the scanner

LANDMARKS = np.array([
    [10,-20],
    [-10,-20],
    [10,20],
    [-10,20],
    [5,10],
    [-5,10],
    [5,-10],
    [-5,-10],
    [0, 5],
    [0, -5]
]) # Landmark locations in x, y (m, m)

# Noise parameters:
Q_STD_DEV = np.array([0.2, np.deg2rad(1)]) # Observation noise - standard deviation
Q = np.diag(Q_STD_DEV) ** 2 # Observation noise - variance
R_STD_DEV = np.array([1, np.deg2rad(10)]) # Prediction noise - standard deviation
R = np.diag(R_STD_DEV) ** 2 # Prediction noise - variance

def get_controls(t: float, noisy: bool=False) -> np.ndarray:
    """
    Gets the control at time t.
    In SLAM research this is in m/s and rad/s
    To simplify my life, I multiply by DT here, so it is the change in m and rad.
    :param t: The current time of the simulation in s
    :param noisy: Adds noise defined in R and R_STD_DEV above if set to True
    :return:
    """
    v = 1
    w = -0.1
    if t<0.5*SIM_TIME:
        w = 0.1
    u = np.array([[v, w]]).T

    if noisy:
        noise = np.random.normal(0, R_STD_DEV)
        noise = np.reshape(noise, u.shape)
        u = u + noise

    u = u * DT
    return u

def motion_model(x: np.ndarray, u: np.ndarray):
    """
    Applies the motion model of a differential drive robot.
    Makes up the prediction stage of the EKF.
    :param x: Input pose
    :param u: Input at time t from get_controls
    :return:
    """
    F = np.eye(3)

    B = np.array([[np.cos(x[2,0]), 0.0],
                  [np.sin(x[2,0]), 0.0],
                  [0.0, 1.0]])

    x = (F @ x) + (B @ u)

    G = np.array([[1, 0, -u[0,0] * np.sin(x[2, 0])],
                  [1, 0, u[0,0] * np.cos(x[2, 0])],
                  [0, 0, 1]])

    return x, G


def pi2pi(angle: float) -> float:
    """
    Wraps an angle provided in radians to be between -pi and pi
    :param angle: Input angle in radians
    :return:
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def make_observations(current_gt_pose: np.ndarray) -> np.ndarray:
    """
    Observes the landmarks defined above. Scanning is limited by the scanner max distance defined above.
    :param current_gt_pose: The robot's current ground truth pose
    :return: np.ndarray of nx2. Each row is [r, theta]
    """
    current_position = np.array([current_gt_pose[0, 0], current_gt_pose[1, 0]]).reshape(-1, 2)
    dx_dy = LANDMARKS - current_position
    ds = np.linalg.norm(dx_dy, axis=1)

    dist_mask = ds <= SCAN_MAX
    thetas = pi2pi(np.arctan2(dx_dy[:,1], dx_dy[:,0])-current_gt_pose[2,0])
    zs = np.array([ds, thetas]).T
    zs = zs[dist_mask]

    noise = np.random.normal(0, Q_STD_DEV, size=zs.shape)
    z_noisy = zs + noise

    return z_noisy

def convert_observations(observations_t: np.ndarray, pose_t: np.ndarray) -> np.ndarray:
    """
    Converts observations from polar in the robot's frame to cartesian coordinates in the world's frame.
    :param observations_t: Observations made at time t. nx2 with each row being (r, theta)
    :param pose_t: Current pose of the robot, either GT or estimated. Form is 3x1 being (x, y, theta)
    :return:
    """
    corrected_angles = observations_t[:, 1] + pose_t[2, 0]
    observations_t_xy = np.array([[pose_t[0, 0] + observations_t[:, 0] * np.cos(corrected_angles),
                         pose_t[1, 0] + observations_t[:, 0] * np.sin(corrected_angles)]])
    return observations_t_xy


def run_simulation(transient_plot: bool=False):
    t = 0
    ts = [t]
    counter = 0

    ground_truth_pose = np.zeros((3,1))
    dead_reckoned_pose = np.zeros((3,1))

    ground_truth_history = np.zeros((1, 3))
    dead_reckoned_history = np.zeros((1, 3))

    if transient_plot:
        plt.figure(figsize=(9, 6))
    while t <= SIM_TIME:
        counter += 1
        t += DT
        ts.append(t)

        u = get_controls(t)
        u_noisy = get_controls(t, True)

        ground_truth_pose, G_t = motion_model(ground_truth_pose, u)
        ground_truth_history = np.vstack((ground_truth_history, ground_truth_pose.T))

        dead_reckoned_pose, G_t_estimated = motion_model(dead_reckoned_pose, u_noisy)
        dead_reckoned_history = np.vstack((dead_reckoned_history, dead_reckoned_pose.T))

        z = make_observations(ground_truth_pose)
        z_xy = convert_observations(z, ground_truth_pose)

        if transient_plot:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_press_event', lambda event: exit(0) if event.key == 'q' else None)

            plt.plot(ground_truth_pose[0], ground_truth_pose[1], ".b", label="Ground truth pose")
            plt.plot(ground_truth_history[:, 0], ground_truth_history[:, 1], "-b")
            plt.plot(dead_reckoned_pose[0], dead_reckoned_pose[1], ".r", label="Dead-reckoned pose")
            plt.plot(dead_reckoned_history[:, 0], dead_reckoned_history[:, 1], "-r")

            plt.plot(LANDMARKS[:,0], LANDMARKS[:,1], "*k", label="Landmarks")
            plt.plot(z_xy[:, 0], z_xy[:, 1], "+g")
            plt.plot([], [], "+g", label="Currently Observed Landmarks")

            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.0001)

    print(f"Done in {counter} iterations")

    plt.figure(figsize=(9, 6))
    plt.plot(ground_truth_history[:, 0], ground_truth_history[:, 1], "-b", label="Ground Truth Pose")
    plt.plot(dead_reckoned_history[:, 0], dead_reckoned_history[:, 1], "-r", label="Dead-reckoned Pose")
    plt.plot(LANDMARKS[:,0], LANDMARKS[:,1], "*k", label="Landmarks")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Simulation output [XY]")

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 6))
    axs[0].plot(ts, ground_truth_history[:, 0], "-b", label="Ground truth x")
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(ts, ground_truth_history[:, 1], "-k", label="Ground truth y")
    axs[1].legend()
    axs[1].grid(True)
    axs[2].plot(ts, ground_truth_history[:, 2], "-r", label="Ground truth θ")
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_xlabel("Time")
    fig.suptitle("Simulation output [XYθ]")
    plt.tight_layout()

    plt.show()


def main():
    run_simulation(True)

if __name__ == "__main__":
    main()
