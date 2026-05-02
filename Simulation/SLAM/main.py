import matplotlib.pyplot as plt
import numpy as np

SIM_TIME = 40*np.pi
DT = 0.2


def get_controls(t: float, ) -> np.ndarray:
    """
    Gets the control at time t.
    In SLAM research this is in m/s and rad/s
    To simplify my life, I multiply by DT here, so it is the change in m and rad.
    :param t: The current time of the simulation in s
    :return:
    """
    v = 1
    w = -0.1
    if t<0.5*SIM_TIME:
        w = 0.1
    u = np.array([[v, w]]).T

    u = u * DT
    return u

def motion_model(x: np.ndarray, u: np.ndarray):
    """
    Applies the motion model of a differential drive robot
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

def main():
    t = 0
    counter = 0

    ground_truth_pose = np.zeros((3,1))
    ground_truth_history = np.zeros((3,1))

    while t <= SIM_TIME:
        print(f"Iteration {counter}")
        counter += 1
        t += DT

        u = get_controls(t)

        ground_truth_pose, G_t = motion_model(ground_truth_pose, u)
        ground_truth_history = np.hstack((ground_truth_history, ground_truth_pose))

        plt.cla()
        # ChatGPT gave me this, exits if you press q
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: exit(0) if event.key == 'q' else None)
        plt.plot(ground_truth_pose[0], ground_truth_pose[1], ".b")

        plt.plot(ground_truth_history[0, :], ground_truth_history[1, :], "-b")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
