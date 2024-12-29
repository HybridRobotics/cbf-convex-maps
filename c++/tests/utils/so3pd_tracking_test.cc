#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>

#include "sccbf/data_types.h"
#include "sccbf/utils/control_utils.h"
#include "sccbf/utils/matrix_utils.h"

namespace cbf = sccbf;

namespace {

struct So3PdInput {
  cbf::MatrixXd R;
  cbf::MatrixXd Rd;
  cbf::VectorXd w;
  cbf::VectorXd wd;
  cbf::VectorXd dwd;
  cbf::MatrixXd inertia;
  double dt;
  double T;
  cbf::So3PdParameters param;
};

void So3TrackingErrors(const So3PdInput& in, cbf::VectorXd& t_seq,
                       cbf::VectorXd& e_R, cbf::VectorXd& e_Omega,
                       cbf::VectorXd& M_norm) {
  cbf::MatrixXd R = in.R;
  const auto Rd = in.Rd;
  cbf::VectorXd w = in.w;
  const auto wd = in.wd;
  const auto dwd = in.dwd;
  const auto inertia = in.inertia;
  const cbf::MatrixXd inertia_inv = inertia.inverse();
  const auto param = in.param;

  double dt = in.dt;
  const auto T = in.T;
  const int N = static_cast<int>(std::ceil(T / dt));
  t_seq = cbf::VectorXd::LinSpaced(N, 0.0, T);
  dt = t_seq(1) - t_seq(0);
  e_R.resize(N);
  e_Omega.resize(N);
  M_norm.resize(N);

  cbf::VectorXd M(3);
  cbf::MatrixXd w_hat(3, 3);
  cbf::VectorXd dw(3);
  for (int i = 0; i < N; ++i) {
    M = cbf::So3PdTrackingControl(R, Rd, w, wd, dwd, inertia, param);

    e_R(i) = 0.5 * (cbf::MatrixXd::Identity(3, 3) - Rd.transpose() * R).trace();
    e_Omega(i) = (w - R.transpose() * Rd * wd).norm();
    M_norm(i) = M.norm();

    cbf::HatMap<3>(w, w_hat);
    dw = inertia_inv * (M - w_hat * inertia * w);
    cbf::IntegrateSo3(R, (w + dw * dt / 2) * dt, R);
    w = w + dw * dt;
  }
}

}  // namespace

int main() {
  // Set inputs.
  const double T = 5.0;
  const double dt = 1e-3;

  const cbf::MatrixXd R = cbf::MatrixXd::Identity(3, 3);
  cbf::MatrixXd Rd(3, 3);
  cbf::RandomRotation(Rd);
  const cbf::VectorXd w = cbf::VectorXd::Random(3);
  const cbf::VectorXd wd = cbf::VectorXd::Zero(3);
  const cbf::VectorXd dwd = cbf::VectorXd::Zero(3);

  const Eigen::Vector3d inertia_diag(2.32 * 1e-3, 2.32 * 1e-3, 4 * 1e-3);
  const cbf::MatrixXd inertia = inertia_diag.asDiagonal();

  const double scale = 1 / 0.0820 * inertia(0, 0);
  const double k_R = 8.81 * scale;
  const double k_Omega = 2.54 * scale;
  const cbf::So3PdParameters param = {k_R, k_Omega};

  const So3PdInput in = {R, Rd, w, wd, dwd, inertia, dt, T, param};

  // Get outputs.
  cbf::VectorXd t_seq, e_R, e_Omega, M_norm;
  So3TrackingErrors(in, t_seq, e_R, e_Omega, M_norm);

  // Save to .csv file.
  std::ofstream outfile("so3pd_tracking_test_data.csv");
  for (int i = 0; i < t_seq.rows(); ++i) {
    outfile << t_seq(i) << "," << e_R(i) << "," << e_Omega(i) << ","
            << M_norm(i) << std::endl;
  }
  outfile.close();

  return EXIT_SUCCESS;
}
