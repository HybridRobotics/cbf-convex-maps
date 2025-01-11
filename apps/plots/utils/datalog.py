import csv
from collections import OrderedDict

import numpy as np


def read_kkt_ode_logs(filename: str) -> dict:
    log = OrderedDict()
    log["t_seq"] = []
    log["x"] = []
    log["solve_time_ode"] = []
    log["solve_time_opt"] = []
    log["dist2_ode"] = []
    log["dist2_opt"] = []
    log["Ddist2_ode"] = []
    log["z_err_norm"] = []
    log["z_opt_norm"] = []
    log["lambda_err_norm"] = []
    log["lambda_opt_norm"] = []
    log["dual_inf_err_norm"] = []
    log["prim_inf_err_norm"] = []
    log["compl_err_norm"] = []

    num_opt_solves = 0
    nr = 0
    data_start = 0
    A, b = np.empty((0, 3)), np.empty((0,))
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for k, row in enumerate(reader):
            # Headers
            if k == 0:
                num_opt_solves = float(row[-1])
                continue
            elif k == 1:
                nr = int(row[-1])
                data_start = 3 + nr
                A, b = np.zeros((nr, 3)), np.zeros((nr,))
                continue
            elif k < data_start:
                if k == data_start - 1:
                    continue
                A[k - 2, :] = np.array([float(row[i]) for i in range(3)])
                b[k - 2] = float(row[-1])
                continue
            # Data
            rowf = [float(value) for value in row]
            for i, key in enumerate(log):
                if i == 0:
                    log[key].append(rowf[i])
                elif i == 1:
                    log[key].append([rowf[j] for j in range(1, 16)])
                else:
                    log[key].append(rowf[i + 14])
    for key in log:
        log[key] = np.array(log[key])
    log["x"] = log["x"].T
    log["num_opt_solves"] = num_opt_solves
    log["A"] = A
    log["b"] = b
    log["t_0"] = log["t_seq"][0]
    log["T"] = log["t_seq"][-1]
    log["dt"] = log["t_seq"][1] - log["t_0"]

    return log


def read_ma_cbf_logs(filename: str) -> dict:
    log = OrderedDict()
    log["t_seq"] = []
    log["x"] = []
    log["solve_time_ode"] = []
    log["solve_time_opt"] = []
    log["solve_time_qp"] = []
    log["dist2_ode"] = []
    log["dist2_opt"] = []
    log["Ddist2_ode"] = []
    log["z_err_norm"] = []
    log["z_opt_norm"] = []
    log["lambda_err_norm"] = []
    log["lambda_opt_norm"] = []

    num_obs = 0
    num_sys = 0
    num_obs_cps = 0
    num_sys_cps = 0
    num_cps = 0
    num_opt_solves = 0
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=",")
        for k, row in enumerate(reader):
            # Headers
            if k == 1:
                num_obs, num_sys = int(row[0]), int(row[1])
                num_obs_cps, num_sys_cps = int(row[2]), int(row[3])
                num_cps = int(row[4])
                num_opt_solves = int(row[5])
                continue
            elif k < 3:
                continue
            # Data
            rowf = [float(value) for value in row]
            for i, key in enumerate(log):
                if i == 0:
                    log[key].append(rowf[i])
                elif i == 1:
                    log[key].append([rowf[j] for j in range(1, 1 + 15 * num_sys)])
                elif i <= 4:
                    idx = 1 + 15 * num_sys
                    log[key].append(rowf[idx + i - 2])
                else:
                    idx = 1 + 15 * num_sys + 3 + (i - 5) * num_cps
                    log[key].append([rowf[j] for j in range(idx, idx + num_cps)])
    for key in log:
        log[key] = np.array(log[key])
    log["x"] = log["x"].T
    log["dist2_ode"] = log["dist2_ode"].T
    log["dist2_opt"] = log["dist2_opt"].T
    log["Ddist2_ode"] = log["Ddist2_ode"].T
    log["z_err_norm"] = log["z_err_norm"].T
    log["z_opt_norm"] = log["z_opt_norm"].T
    log["lambda_err_norm"] = log["lambda_err_norm"].T
    log["lambda_opt_norm"] = log["lambda_opt_norm"].T
    log["num_obs"] = num_obs
    log["num_sys"] = num_sys
    log["num_obs_cps"] = num_obs_cps
    log["num_sys_cps"] = num_sys_cps
    log["num_cps"] = num_cps
    log["num_opt_solves"] = num_opt_solves
    log["t_0"] = log["t_seq"][0]
    log["T"] = log["t_seq"][-1]
    log["dt"] = log["t_seq"][1] - log["t_0"]

    return log
