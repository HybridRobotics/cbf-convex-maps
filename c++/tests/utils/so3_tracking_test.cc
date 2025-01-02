#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>

#include "sccbf/data_types.h"
#include "sccbf/utils/control_utils.h"
#include "sccbf/utils/matrix_utils.h"

using namespace sccbf;

namespace {

struct So3PdInput {
  MatrixXd R;
  MatrixXd Rd;
  VectorXd w;
  VectorXd wd;
  VectorXd dwd;
  MatrixXd inertia;
  double dt;
  double T;
  So3PdParameters param;
};

void So3PTrackingErrors(const So3PdInput& in, VectorXd& t_seq, VectorXd& e_R,
                        VectorXd& w_norm) {
  MatrixXd R = in.R;
  const auto Rd = in.Rd;
  const auto wd = in.wd;
  const auto param = in.param;

  double dt = in.dt;
  const auto T = in.T;
  const int N = static_cast<int>(std::ceil(T / dt));
  t_seq = VectorXd::LinSpaced(N, 0.0, T);
  dt = t_seq(1) - t_seq(0);
  e_R.resize(N);
  w_norm.resize(N);

  VectorXd w(3);
  for (int i = 0; i < N; ++i) {
    w = So3PTrackingControl(R, Rd, wd, param);

    e_R(i) = 0.5 * (MatrixXd::Identity(3, 3) - Rd.transpose() * R).trace();
    w_norm(i) = w.norm();

    IntegrateSo3(R, w * dt, R);
  }
}

void So3PdTrackingErrors(const So3PdInput& in, VectorXd& t_seq, VectorXd& e_R,
                         VectorXd& e_Omega, VectorXd& M_norm) {
  MatrixXd R = in.R;
  const auto Rd = in.Rd;
  VectorXd w = in.w;
  const auto wd = in.wd;
  const auto dwd = in.dwd;
  const auto inertia = in.inertia;
  const MatrixXd inertia_inv = inertia.inverse();
  const auto param = in.param;

  double dt = in.dt;
  const auto T = in.T;
  const int N = static_cast<int>(std::ceil(T / dt));
  t_seq = VectorXd::LinSpaced(N, 0.0, T);
  dt = t_seq(1) - t_seq(0);
  e_R.resize(N);
  e_Omega.resize(N);
  M_norm.resize(N);

  VectorXd M(3);
  MatrixXd w_hat(3, 3);
  VectorXd dw(3);
  for (int i = 0; i < N; ++i) {
    M = So3PdTrackingControl(R, Rd, w, wd, dwd, inertia, param);

    e_R(i) = 0.5 * (MatrixXd::Identity(3, 3) - Rd.transpose() * R).trace();
    e_Omega(i) = (w - R.transpose() * Rd * wd).norm();
    M_norm(i) = M.norm();

    HatMap<3>(w, w_hat);
    dw = inertia_inv * (M - w_hat * inertia * w);
    IntegrateSo3(R, (w + dw * dt / 2) * dt, R);
    w = w + dw * dt;
  }
}

}  // namespace

int main() {
  // Set inputs.
  const double T = 5.0;
  const double dt = 1e-3;

  const MatrixXd R = MatrixXd::Identity(3, 3);
  MatrixXd Rd(3, 3);
  RandomRotation(Rd);
  const VectorXd w = VectorXd::Random(3);
  const VectorXd wd = VectorXd::Zero(3);
  const VectorXd dwd = VectorXd::Zero(3);

  const Eigen::Vector3d inertia_diag(2.32 * 1e-3, 2.32 * 1e-3, 4 * 1e-3);
  const MatrixXd inertia = inertia_diag.asDiagonal();

  const double scale = 1 / 0.0820 * inertia(0, 0);
  const double k_R = 8.81 * scale;
  const double k_Omega = 2.54 * scale;
  const So3PdParameters param = {k_R, k_Omega};

  const So3PdInput in = {R, Rd, w, wd, dwd, inertia, dt, T, param};

  // Get first-order tracking data.
  {
    So3PdInput in_(in);
    in_.param.k_R = 3.5;
    VectorXd t_seq, e_R, w_norm;
    So3PTrackingErrors(in_, t_seq, e_R, w_norm);

    // Save to .csv file.
    std::ofstream outfile("so3p_tracking_test_data.csv");
    for (int i = 0; i < t_seq.rows(); ++i) {
      outfile << t_seq(i) << "," << e_R(i) << "," << w_norm(i) << std::endl;
    }
    outfile.close();
  }

  // Get second-order tracking data.
  {
    VectorXd t_seq, e_R, e_Omega, M_norm;
    So3PdTrackingErrors(in, t_seq, e_R, e_Omega, M_norm);

    // Save to .csv file.
    std::ofstream outfile("so3pd_tracking_test_data.csv");
    for (int i = 0; i < t_seq.rows(); ++i) {
      outfile << t_seq(i) << "," << e_R(i) << "," << e_Omega(i) << ","
              << M_norm(i) << std::endl;
    }
    outfile.close();
  }

  return EXIT_SUCCESS;
}
