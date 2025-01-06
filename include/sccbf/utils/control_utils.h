#ifndef SCCBF_UTILS_CONTROL_UTILS_H_
#define SCCBF_UTILS_CONTROL_UTILS_H_

#include <Eigen/Core>
#include <cassert>

#include "sccbf/data_types.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

struct So3PdParameters {
  double k_R;
  double k_Omega;
};

inline VectorXd So3PTrackingControl(const MatrixXd& R, const MatrixXd& Rd,
                                    const VectorXd& wd,
                                    const So3PdParameters& param) {
  assert((R.rows() == 3) && (R.cols() == 3));
  assert((Rd.rows() == 3) && (Rd.cols() == 3));
  assert(wd.rows() == 3);
  const double k_R = param.k_R;
  assert(k_R > 0);

  const auto R_err = Rd.transpose() * R;
  const auto e_R_hat = 0.5 * (R_err - R_err.transpose());
  const Eigen::Vector3d e_R(e_R_hat(2, 1), e_R_hat(0, 2), e_R_hat(1, 0));

  const VectorXd w = R_err.transpose() * wd - k_R * e_R;
  return w;
}

inline VectorXd So3PdTrackingControl(const MatrixXd& R, const MatrixXd& Rd,
                                     const VectorXd& w, const VectorXd& wd,
                                     const VectorXd& dwd,
                                     const MatrixXd& inertia,
                                     const So3PdParameters& param) {
  assert((R.rows() == 3) && (R.cols() == 3));
  assert((Rd.rows() == 3) && (Rd.cols() == 3));
  assert((w.rows() == 3) && (wd.rows() == 3) && (dwd.rows() == 3));
  assert((inertia.rows() == 3) && (inertia.cols() == 3));
  const double k_R = param.k_R;
  const double k_Omega = param.k_Omega;
  assert((k_R > 0) && (k_Omega > 0));

  MatrixXd w_hat = MatrixXd::Zero(3, 3);
  HatMap<3>(w, w_hat);
  const auto e_R_hat = 0.5 * (Rd.transpose() * R - R.transpose() * Rd);
  const Eigen::Vector3d e_R(e_R_hat(2, 1), e_R_hat(0, 2), e_R_hat(1, 0));
  const auto e_Omega = w - R.transpose() * Rd * wd;

  const VectorXd M =
      -k_R * e_R - k_Omega * e_Omega + w_hat * inertia * w -
      inertia * (w_hat * R.transpose() * Rd * wd - R.transpose() * Rd * dwd);
  return M;
}

}  // namespace sccbf

#endif  // SCCBF_UTILS_CONTROL_UTILS_H_
