import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from utils.datalog import read_ma_cbf_logs

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"

_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def _print_statistics(log: dict) -> None:
    # Fraction of ipopt solves
    num_opt_solves = int(log["num_opt_solves"])
    frac_opt = num_opt_solves / (len(log["t_seq"]) * log["num_cps"])

    num_cps = log["num_cps"]
    print(f"Number of collision pairs: {num_cps:5d}")

    # Solution times.
    avg_solve_time_opt = np.mean(log["solve_time_opt"]) * 1e3
    print(f"Solution time (opt) (ms): avg {avg_solve_time_opt:8.5f}")

    solve_time_qp = log["solve_time_qp"]
    avg_solve_time_qp = np.mean(solve_time_qp) * 1e3
    std_solve_time_qp = np.std(solve_time_qp) * 1e3
    p50_solve_time_qp = np.percentile(solve_time_qp, 50) * 1e3
    p90_solve_time_qp = np.percentile(solve_time_qp, 90) * 1e3
    p99_solve_time_qp = np.percentile(solve_time_qp, 99) * 1e3
    print(f"Solution time (qp) (ms): avg {avg_solve_time_qp:8.5f}, ", end="")
    print(f"std {std_solve_time_qp:8.5f}, ", end="")
    print(f"p50 {p50_solve_time_qp:8.5f}, ", end="")
    print(f"p90 {p90_solve_time_qp:8.5f}, ", end="")
    print(f"p99 {p99_solve_time_qp:8.5f}")

    solve_time_ode = log["solve_time_ode"][
        log["solve_time_ode"] * 1e3 < 0.6
    ]  # from histogram.
    avg_solve_time_ode = np.mean(solve_time_ode) * 1e3
    std_solve_time_ode = np.std(solve_time_ode) * 1e3
    p50_solve_time_ode = np.percentile(solve_time_ode, 50) * 1e3
    p90_solve_time_ode = np.percentile(solve_time_ode, 90) * 1e3
    p99_solve_time_ode = np.percentile(solve_time_ode, 99) * 1e3
    print(f"Solution time (ode) (ms): avg {avg_solve_time_ode:8.5f}, ", end="")
    print(f"std {std_solve_time_ode:8.5f}, ", end="")
    print(f"p50 {p50_solve_time_ode:8.5f}, ", end="")
    print(f"p90 {p90_solve_time_ode:8.5f}, ", end="")
    print(f"p99 {p99_solve_time_ode:8.5f}")

    # Minimum distance error.
    dist_opt = np.sqrt(log["dist2_opt"])
    dist_ode = np.sqrt(log["dist2_ode"])
    dist_err = np.abs(dist_opt - dist_ode)
    dist_err_max = np.max(dist_err)
    dist_err_rel = np.max(dist_err / dist_opt)
    print(f"Distance error (max) (m)     : {dist_err_max:8.5f}")
    print(f"Distance relative error (max): {dist_err_rel:8.5f}")

    # Minimum distance derivative.
    Ddist2_opt = np.gradient(log["dist2_opt"], log["t_seq"], axis=1)
    Ddist_ode = log["Ddist2_ode"] / (2 * dist_ode)
    Ddist_ode_max = np.max(Ddist_ode)
    Ddist_err = np.abs(Ddist2_opt / (2 * dist_opt) - Ddist_ode)
    avg_Ddist_err, std_Ddist_err = np.mean(Ddist_err), np.std(Ddist_err)
    p50_Ddist_err, p90_Ddist_err, p99_Ddist_err = np.percentile(Ddist_err, [50, 90, 99])
    print(f"Distance derivative error (m/s)      : Avg {avg_Ddist_err:8.5f}, ", end="")
    print(f"std {std_Ddist_err:8.5f}, ", end="")
    print(f"p50 {p50_Ddist_err:8.5f}, ", end="")
    print(f"p90 {p90_Ddist_err:8.5f}, ", end="")
    print(f"p99 {p99_Ddist_err:8.5f}")
    print(f"Distance derivative (max) (ode) (m/s): {Ddist_ode_max:8.5f}")

    # Primal and dual solution errors.
    z_rel_err_max = np.max(log["z_err_norm"] / log["z_opt_norm"])
    print(f"Primal solution relative error (max): {z_rel_err_max:8.5f}")
    lambda_rel_err_max = np.max(log["lambda_err_norm"] / log["lambda_opt_norm"])
    print(f"Dual solution relative error (max)  : {lambda_rel_err_max:8.5f}")

    # Fraction of ipopt solves.
    print(f"Number of ipopt solves  : {num_opt_solves:5d}")
    print(f"Fraction of ipopt solves: {frac_opt:5.3f}")


def _plot_data(log: dict):
    fig_width = 2.57  # [inch].
    fig_height = 1.75  # [inch].
    fig_dpi = 200
    save_dpi = 1000  # >= 600
    save_fig = False
    font_size = 8  # [pt].
    font_dict = {"fontsize": font_size, "fontstyle": "normal", "fontweight": "normal"}
    axis_margins = 0.05

    # Data.
    t_seq = log["t_seq"]
    Tf = 80  # [s]
    Nf = int(Tf / 100.0 * len(t_seq) - 1)
    dist_opt = np.sqrt(log["dist2_opt"])
    dist_opt_avg = np.mean(dist_opt, axis=0)
    dist_opt_min = np.min(dist_opt, axis=0)
    dist_opt_max = np.max(dist_opt, axis=0)
    margin_min = 0.2  # [m].
    z_rel_err = log["z_err_norm"] / log["z_opt_norm"]
    z_rel_err_avg = np.mean(z_rel_err, axis=0)
    z_rel_err_min = np.min(z_rel_err, axis=0)
    z_rel_err_max = np.max(z_rel_err, axis=0)
    lambda_rel_err = log["lambda_err_norm"] / log["lambda_opt_norm"]
    lambda_rel_err_avg = np.mean(lambda_rel_err, axis=0)
    lambda_rel_err_min = np.min(lambda_rel_err, axis=0)
    lambda_rel_err_max = np.max(lambda_rel_err, axis=0)
    dist_ode = np.sqrt(log["dist2_ode"])
    dist_rel_err = np.abs(dist_opt - dist_ode) / dist_opt
    dist_rel_err_avg = np.mean(dist_rel_err, axis=0)
    dist_rel_err_min = np.min(dist_rel_err, axis=0)
    dist_rel_err_max = np.max(dist_rel_err, axis=0)
    Ddist_opt = np.gradient(dist_opt, t_seq, axis=1)
    Ddist_ode = log["Ddist2_ode"] / (2 * dist_ode)
    Ddist_err = Ddist_opt - Ddist_ode
    Ddist_err_avg = np.mean(Ddist_err, axis=0)
    Ddist_err_min = np.min(Ddist_err, axis=0)
    Ddist_err_max = np.max(Ddist_err, axis=0)

    ## Plot 1: Minimum distance.
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
    ax.plot(t_seq[:Nf:sp], dist_opt_avg[:Nf:sp], "-b", lw=0.75, label=r"$h(t)$")
    ax.fill_between(
        t_seq[:Nf:sp],
        dist_opt_min[:Nf:sp],
        dist_opt_max[:Nf:sp],
        color="b",
        alpha=0.25,
        ec="none",
    )
    ax.plot(
        t_seq[:Nf:sp],
        margin_min * np.ones_like(t_seq[:Nf:sp]),
        "--r",
        lw=0.75,
        label=r"$\epsilon_\text{dist}$",
    )
    # Plot properties.
    ax.grid(axis="y", lw=0.25, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ $(s)$", **font_dict)
    ax.set_ylabel(r"minimum distance $(m)$", **font_dict)
    ax.legend(
        loc="center right",
        fontsize=font_size,
        framealpha=0.5,
        fancybox=False,
        edgecolor="black",
    )
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    # ax.set_xticks([])
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=3))
    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10, subs=np.linspace(0.2, 1, 5), numticks=50)
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.margins(axis_margins, axis_margins)
    # ax.set_ylim([])

    plt.show()

    if save_fig:
        fig.savefig(_DIR_PATH + "/cbf_dist.png", dpi=save_dpi)  # , bbox_inches='tight')

    ## Plot 2: KKT error.
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
    # ax.plot(
    #     t_seq[:Nf:sp],
    #     z_rel_err_avg[:Nf:sp] + 1e-5,
    #     "-b",
    #     lw=1,
    #     label=r"$|\Delta z^*(t)|/|z^*(t)|$",
    # )
    # ax.fill_between(
    #     t_seq[:Nf:sp],
    #     z_rel_err_min[:Nf:sp] + 1e-5,
    #     z_rel_err_max[:Nf:sp] + 1e-5,
    #     color="b",
    #     alpha=0.25,
    #     ec="none",
    # )
    ax.plot(
        t_seq[:Nf:sp],
        z_rel_err_max[:Nf:sp] + 1e-5,
        "-b",
        lw=0.75,
        label=r"$|\Delta z^*(t)|/|z^*(t)|$",
    )
    # ax.plot(
    #     t_seq[:Nf:sp],
    #     lambda_rel_err_avg[:Nf:sp] + 1e-5,
    #     "-.r",
    #     lw=1,
    #     label=r"$|\Delta \lambda^*(t)|/|\lambda^*(t)|$",
    # )
    # ax.fill_between(
    #     t_seq[:Nf:sp],
    #     lambda_rel_err_min[:Nf:sp] + 1e-5,
    #     lambda_rel_err_max[:Nf:sp] + 1e-5,
    #     color="r",
    #     alpha=0.25,
    #     ec="none",
    # )
    ax.plot(
        t_seq[:Nf:sp],
        lambda_rel_err_max[:Nf:sp] + 1e-5,
        "-.r",
        lw=0.75,
        label=r"$|\Delta \lambda^*(t)|/|\lambda^*(t)|$",
    )
    # Plot properties.
    ax.grid(axis="y", lw=0.25, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ $(s)$", **font_dict)
    ax.set_ylabel(r"max. relative KKT error", **font_dict)
    ax.legend(
        loc="upper right",
        fontsize=font_size,
        framealpha=1.0,
        fancybox=False,
        edgecolor="black",
    )
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    # ax.set_xticks([])
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=3))
    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10, subs=np.linspace(0.2, 1, 5), numticks=50)
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.margins(axis_margins, axis_margins)
    ax.set_ylim([2e-5, 2e-1])

    plt.show()

    if save_fig:
        fig.savefig(
            _DIR_PATH + "/cbf_kkt_err.png", dpi=save_dpi
        )  # , bbox_inches='tight')

    ## Plot 3: Minimum distance error.
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
    ax.plot(
        t_seq[:Nf:sp],
        dist_rel_err_avg[:Nf:sp] + 1e-5,
        "-b",
        lw=0.75,
        label=r"$\Delta h(t)/h(t)$",
    )
    ax.fill_between(
        t_seq[:Nf:sp],
        dist_rel_err_min[:Nf:sp] + 1e-5,
        dist_rel_err_max[:Nf:sp] + 1e-5,
        color="b",
        alpha=0.25,
        ec="none",
    )
    # Plot properties.
    ax.grid(axis="y", lw=0.25, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ $(s)$", **font_dict)
    ax.set_ylabel(r"relative distance error", **font_dict)
    ax.legend(
        loc="upper right",
        fontsize=font_size,
        framealpha=1.0,
        fancybox=False,
        edgecolor="black",
    )
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    # ax.set_xticks([])
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=3))
    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10, subs=np.linspace(0.2, 1, 5), numticks=50)
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.margins(axis_margins, axis_margins)
    ax.set_ylim([1e-5, 3e-3])

    plt.show()

    if save_fig:
        fig.savefig(
            _DIR_PATH + "/cbf_dist_err.png", dpi=save_dpi
        )  # , bbox_inches='tight')

    ## Plot 4: Minimum distance Derivative error (Inaccurate comparison due to sampling).
    if False:
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
        ax.plot(t_seq[::sp], np.zeros_like(t_seq[::sp]), "--r", lw=1)
        ax.plot(
            t_seq[::sp], Ddist_err_avg[::sp], "-b", lw=1, label=r"$\Delta \dot{h}(t)$"
        )
        ax.fill_between(
            t_seq[::sp],
            Ddist_err_min[::sp],
            Ddist_err_max[::sp],
            color="b",
            alpha=0.25,
            ec="none",
        )
        # Plot properties.
        ax.grid(axis="y", lw=0.25, alpha=0.5)
        ax.set_xlabel(r"$t$ $(s)$", **font_dict)
        ax.set_ylabel(r"error $(m/s)$", **font_dict)
        ax.legend(
            loc="upper right",
            fontsize=font_size,
            framealpha=0.5,
            fancybox=False,
            edgecolor="black",
        )
        ax.tick_params(axis="both", which="major", labelsize=font_size)
        ax.margins(axis_margins, axis_margins)
        # ax.set_ylim([])

        plt.show()

        if save_fig:
            fig.savefig(
                _DIR_PATH + "/cbf_Ddist_err.png", dpi=save_dpi
            )  # , bbox_inches='tight')

    # x = log["x"]
    # speed = np.linalg.norm(x[3:6, :], axis=0)
    # quad_angle = x[-1, :]
    # plt.plot(speed)
    # plt.plot(quad_angle)
    # plt.show()


if __name__ == "__main__":
    relative_path = "/../../build"

    filename = _DIR_PATH + relative_path + "/ma_cbf_data.csv"
    log = read_ma_cbf_logs(filename)
    _print_statistics(log)
    _plot_data(log)
