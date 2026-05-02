import numpy as np
import matplotlib.pyplot as plt

def single_plot():
    x = np.arange(0, 2*np.pi+0.01, 0.01)
    y = np.sin(x)

    plt.figure(figsize=(9,6))
    plt.plot(x, y, label="sin(x)", linewidth=2)
    plt.title("sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.xticks(
        [0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi],
        [r'$0$', r'$0.5\pi$', r'$\pi$', r'$1.5\pi$', r'$2\pi$']
    )

    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.legend()
    plt.xlim(min(x), max(x))
    plt.ylim(1.05*min(y), 1.05*max(y))
    plt.grid(True)
    plt.show()

def multiple_plot():
    x = np.arange(0, 2 * np.pi + 0.01, 0.01)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    fig, (ax1,ax2) = plt.subplots(2,1, sharex='all', figsize=(9,6))

    ax1.plot(x, y_sin, label="sin(x)")
    ax1.set_title("y=sin(x)")
    ax1.grid(True)

    ax2.plot(x, y_cos, label="sin(x)")
    ax2.set_title("y=cos(x)")

    plt.xticks(
        [0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi],
        [r'$0$', r'$0.5\pi$', r'$\pi$', r'$1.5\pi$', r'$2\pi$']
    )
    plt.xlim(min(x), max(x))

    plt.suptitle("sin(x) and cos(x)")
    plt.show()


multiple_plot()
