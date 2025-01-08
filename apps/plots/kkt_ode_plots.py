import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from utils.datalog import read_kkt_ode_logs

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"

_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def _print_statistics(log: dict) -> None:
    # Solution times.
    avg_solve_time_opt = np.mean(log["solve_time_opt"]) * 1e3
    print(f"Solution time (opt) (ms): avg {avg_solve_time_opt:7.3f}")
    avg_solve_time_ode = np.mean(log["solve_time_ode"]) * 1e3
    std_solve_time_ode = np.std(log["solve_time_ode"]) * 1e3
    p50_solve_time_ode = np.percentile(log["solve_time_ode"], 50) * 1e3
    p90_solve_time_ode = np.percentile(log["solve_time_ode"], 90) * 1e3
    p99_solve_time_ode = np.percentile(log["solve_time_ode"], 99) * 1e3
    print(f"Solution time (ode) (ms): avg {avg_solve_time_ode:7.3f}, ", end="")
    print(f"std {std_solve_time_ode:7.3f}, ", end="")
    print(f"p50 {p50_solve_time_ode:7.3f}, ", end="")
    print(f"p90 {p90_solve_time_ode:7.3f}, ", end="")
    print(f"p99 {p99_solve_time_ode:7.3f}")

    # Minimum distance error.
    dist_opt = np.sqrt(log["dist2_opt"])
    dist_ode = np.sqrt(log["dist2_ode"])
    dist_err = np.abs(dist_opt - dist_ode)
    dist_err_max = np.max(dist_err)
    dist_err_rel = np.max(dist_err / dist_opt)
    print(f"Distance error (max) (m)     : {dist_err_max:6.3f}")
    print(f"Distance relative error (max): {dist_err_rel:6.3f}")

    # Minimum distance derivative.
    Ddist2_opt = np.gradient(log["dist2_opt"], log["t_seq"])
    Ddist_ode = log["Ddist2_ode"] / (2 * dist_ode)
    Ddist_ode_max = np.max(Ddist_ode)
    Ddist_err = np.abs(Ddist2_opt / (2 * dist_opt) - Ddist_ode)
    avg_Ddist_err, std_Ddist_err = np.mean(Ddist_err), np.std(Ddist_err)
    p50_Ddist_err, p90_Ddist_err, p99_Ddist_err = np.percentile(Ddist_err, [50, 90, 99])
    print(f"Distance derivative error (m/s)      : Avg {avg_Ddist_err:7.3f}, ", end="")
    print(f"std {std_Ddist_err:7.3f}, ", end="")
    print(f"p50 {p50_Ddist_err:7.3f}, ", end="")
    print(f"p90 {p90_Ddist_err:7.3f}, ", end="")
    print(f"p99 {p99_Ddist_err:7.3f}")
    print(f"Distance derivative (max) (ode) (m/s): {Ddist_ode_max:7.3f}")

    # Primal and dual solution errors.
    z_rel_err_max = np.max(log["z_err_norm"] / log["z_opt_norm"])
    print(f"Primal solution relative error (max): {z_rel_err_max:6.3f}")
    lambda_rel_err_max = np.max(log["lambda_err_norm"] / log["lambda_opt_norm"])
    print(f"Dual solution relative error (max)  : {lambda_rel_err_max:6.3f}")

    # KKT errors.
    dual_inf_err_max = np.max(log["dual_inf_err_norm"])
    prim_inf_err_max = np.max(log["prim_inf_err_norm"])
    compl_err_max = np.max(log["compl_err_norm"])
    print(f"Dual infeasibility error (max)  : {dual_inf_err_max:7.3f}")
    print(f"Primal infeasibility error (max): {prim_inf_err_max:7.3f}")
    print(f"Complementarity error (max)     : {compl_err_max:7.3f}")

    # Fraction of ipopt solves.
    num_opt_solves = int(log["num_opt_solves"])
    Nlog = len(log["t_seq"])
    frac_opt = num_opt_solves / Nlog
    print(f"Number of solution steps: {Nlog:5d}")
    print(f"Number of ipopt solves  : {num_opt_solves:5d}")
    print(f"Fraction of ipopt solves: {frac_opt:5.3f}")

    # Tracking error.
    scale = 2.0
    radius = 6.0  # [m].
    T_xy = 20.0 * scale  # [s].
    height = 3.0  # [m].
    T_z = 11.0 * scale  # [s].
    f_xy = 2 * np.pi / T_xy
    f_z = 2 * np.pi / T_z
    t_seq = log["t_seq"]
    pd = np.empty((3, len(t_seq)))
    pd[0, :] = radius * np.cos(f_xy * t_seq)
    pd[1, :] = radius * np.sin(f_xy * t_seq)
    pd[2, :] = height * np.sin(f_z * t_seq)
    vd = np.empty((3, len(t_seq)))
    vd[0, :] = -radius * f_xy * np.sin(f_xy * t_seq)
    vd[1, :] = radius * f_xy * np.cos(f_xy * t_seq)
    vd[2, :] = height * f_z * np.cos(f_z * t_seq)
    x = log["x"]
    p, v = x[0:3, :], x[3:6, :]
    p_err = np.linalg.norm(p - pd, axis=0)
    v_err = np.linalg.norm(v - vd, axis=0)

    # plt.plot(t_seq, p_err, "-b")
    # plt.plot(t_seq, v_err, "-r")
    # plt.show()


def _plot_data(log: dict):
    fig_width = 2.57  # 3.54  # [inch].
    fig_height = 1.75  # [inch].
    fig_dpi = 200
    save_dpi = 1000  # >= 600
    save_fig = False
    font_size = 8  # [pt].
    font_dict = {"fontsize": font_size, "fontstyle": "normal", "fontweight": "normal"}
    axis_margins = 0.05

    # Data.
    t_seq = log["t_seq"]
    z_rel_err_max = log["z_err_norm"] / log["z_opt_norm"]
    lambda_rel_err_max = log["lambda_err_norm"] / log["lambda_opt_norm"]
    dist_opt = np.sqrt(log["dist2_opt"])
    dist_ode = np.sqrt(log["dist2_ode"])
    dist_rel_err = np.abs(dist_opt - dist_ode) / dist_opt
    # Ddist_opt = np.gradient(log["dist2_opt"], t_seq) / (2 * dist_opt)
    Ddist_opt = np.gradient(dist_opt, t_seq)
    Ddist_ode = log["Ddist2_ode"] / (2 * dist_ode)
    Ddist_err = Ddist_opt - Ddist_ode

    ## Plot 1: KKT error.
    sp = 21
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(fig_width, fig_height),
        dpi=fig_dpi,
        sharey=False,
        layout="constrained",
    )
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    # Line plots.
    ax.plot(
        t_seq[::sp],
        lambda_rel_err_max[::sp],
        "-.r",
        lw=0.75,
        label=r"$|\Delta \lambda^*(t)|/|\lambda^*(t)|$",
    )
    ax.plot(
        t_seq[::sp],
        z_rel_err_max[::sp],
        "-b",
        lw=0.75,
        label=r"$|\Delta z^*(t)|/|z^*(t)|$",
    )
    # Plot properties.
    ax.grid(axis="y", lw=0.25, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ $(s)$", **font_dict)
    ax.set_ylabel(r"relative KKT error", **font_dict)
    leg = ax.legend(
        loc="upper right",
        fontsize=font_size,
        framealpha=1.0,
        fancybox=False,
        edgecolor="black",
    )
    leg.get_frame().set_linewidth(0.75)
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    # ax.set_xticks([])
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=3))
    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10, subs=np.linspace(0.2, 1, 5), numticks=50)
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.margins(axis_margins, axis_margins)
    ax.set_ylim([5e-5, 2e-3])

    plt.show()

    if save_fig:
        fig.savefig(
            _DIR_PATH + "/kkt_ode_kkt_err.png", dpi=save_dpi
        )  # , bbox_inches='tight')

    ## Plot 2: Minimum distance error.
    sp = 20
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(fig_width, fig_height),
        dpi=fig_dpi,
        sharey=False,
        layout="constrained",
    )
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    # Line plots.
    ax.plot(t_seq[::sp], dist_rel_err[::sp], "-b", lw=0.75, label=r"$\Delta h(t)/h(t)$")
    # Plot properties.
    ax.grid(axis="y", lw=0.25, alpha=0.5)
    ax.set_xlabel(r"$t$ $(s)$", **font_dict)
    ax.set_ylabel(r"relative distance error", **font_dict)
    leg = ax.legend(
        loc="upper left",
        fontsize=font_size,
        framealpha=1.0,
        fancybox=False,
        edgecolor="black",
    )
    leg.get_frame().set_linewidth(0.75)
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(1e-2, 0.0))
    ax.margins(axis_margins, axis_margins)
    ax.set_ylim([0, 1e-4])

    plt.show()

    if save_fig:
        fig.savefig(
            _DIR_PATH + "/kkt_ode_dist_err.png", dpi=save_dpi
        )  # , bbox_inches='tight')

    ## Plot 3: Minimum distance Derivative error.
    sp = 23
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(fig_width, fig_height),
        dpi=fig_dpi,
        sharey=False,
        layout="constrained",
    )
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    # Line plots.
    # ax.plot(t_seq[::sp], np.zeros_like(t_seq[::sp]), "--r", lw=0.75)
    ax.plot(t_seq[::sp], Ddist_err[::sp], "-b", lw=0.75, label=r"$\Delta \dot{h}(t)$")
    # Plot properties.
    ax.grid(axis="y", lw=0.25, alpha=0.5)
    ax.set_xlabel(r"$t$ $(s)$", **font_dict)
    ax.set_ylabel(r"distance derivative" "\n" r"error $(m/s)$", **font_dict)
    leg = ax.legend(
        loc="upper right",
        fontsize=font_size,
        framealpha=1.0,
        fancybox=False,
        edgecolor="black",
    )
    leg.get_frame().set_linewidth(0.75)
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(1e-2, 0.0))
    ax.margins(axis_margins, axis_margins)
    # ax.set_ylim([-1e-2, 5e-2])

    plt.show()

    if save_fig:
        fig.savefig(
            _DIR_PATH + "/kkt_ode_Ddist_err.png", dpi=save_dpi
        )  # , bbox_inches='tight')


if __name__ == "__main__":
    relative_path = "/../../build"
    type = 0  # 0, 1, or 2.

    filename = _DIR_PATH + relative_path + "/kkt_ode_data_" + str(type) + ".csv"
    log = read_kkt_ode_logs(filename)
    _print_statistics(log)
    _plot_data(log)
