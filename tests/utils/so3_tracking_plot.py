import csv
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"

_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def _plot_so3p_tracking_data(filename: str):
    # Read .csv data.
    t_seq = []
    e_R = []
    w_norm = []
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            rowf = [float(value) for value in row]
            t_seq.append(rowf[0])
            e_R.append(rowf[1])
            w_norm.append(rowf[2])
    t_seq = np.array(t_seq)
    e_R = np.array(e_R)
    w_norm = np.array(w_norm)

    T: float = t_seq[-1]
    dt: float = t_seq[1] - t_seq[0]

    # Plot data.
    fig_width, fig_height = 3.54, 5  # [in].
    _, ax = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        dpi=200,
        sharex=True,
        layout="constrained",
    )
    t_seq = np.arange(0, T, dt)
    ax[0].plot(t_seq, e_R, "-b", lw=1)
    ax[0].set_ylabel(r"$e_R(R, R_d)$")
    ax[1].plot(t_seq, w_norm, "-b", lw=1)
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"$\Vert w\Vert_2$")

    plt.show()


def _plot_so3pd_tracking_data(filename: str):
    # Read .csv data.
    t_seq = []
    e_R = []
    e_Omega = []
    M_norm = []
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            rowf = [float(value) for value in row]
            t_seq.append(rowf[0])
            e_R.append(rowf[1])
            e_Omega.append(rowf[2])
            M_norm.append(rowf[3])
    t_seq = np.array(t_seq)
    e_R = np.array(e_R)
    e_Omega = np.array(e_Omega)
    M_norm = np.array(M_norm)

    T: float = t_seq[-1]
    dt: float = t_seq[1] - t_seq[0]

    # Plot data.
    fig_width, fig_height = 3.54, 5  # [in].
    _, ax = plt.subplots(
        3,
        1,
        figsize=(fig_width, fig_height),
        dpi=200,
        sharex=True,
        layout="constrained",
    )
    t_seq = np.arange(0, T, dt)
    ax[0].plot(t_seq, e_R, "-b", lw=1)
    ax[0].set_ylabel(r"$e_R(R, R_d)$")
    ax[1].plot(t_seq, e_Omega, "-b", lw=1)
    ax[1].set_ylabel(r"$e_\Omega(\omega, \omega_d)$")
    ax[2].plot(t_seq, M_norm, "-b", lw=1)
    ax[2].set_xlabel(r"$t$")
    ax[2].set_ylabel(r"$\Vert M\Vert_2$")

    plt.show()


if __name__ == "__main__":
    relative_path = "/../../build"

    filename = _DIR_PATH + relative_path + "/so3pd_tracking_test_data.csv"
    _plot_so3pd_tracking_data(filename)

    filename = _DIR_PATH + relative_path + "/so3p_tracking_test_data.csv"
    _plot_so3p_tracking_data(filename)
