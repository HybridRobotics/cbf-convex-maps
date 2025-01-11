#include "sccbf/geometry/quadrotor_uncertainty.h"

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

QuadrotorUncertainty::QuadrotorUncertainty(const MatrixXd& Q,
                                           const Eigen::Vector4d& coeff,
                                           double margin)
    : ConvexSet(kNz, kNr, kNx, kNdx, margin), Q_(Q), coeff_(coeff) {
  assert(coeff(0) > std::abs(coeff(1)));
  assert((Q.rows() == kNz) && (Q.cols() == kNz));
  if (!Q.isApprox(Q.transpose())) {
    throw std::runtime_error("Q is not symmetric!");
  }
  if (!IsPositiveDefinite(Q)) {
    throw std::runtime_error("Q is not positive definite!");
  }

  CheckDimensions();
}

QuadrotorUncertainty::~QuadrotorUncertainty() {}

const Derivatives& QuadrotorUncertainty::UpdateDerivatives(
    const VectorXd& x, const VectorXd& dx, const VectorXd& z, const VectorXd& y,
    DerivativeFlags flag) {
  assert(x.rows() == kNx);
  assert(dx.rows() == kNdx);
  assert(z.rows() == kNz);
  assert(y.rows() == kNr);

  const double kPi = static_cast<double>(EIGEN_PI);

  const auto p = x.head<3>();
  const auto R = x.tail<9>().reshaped(3, 3);
  // const auto zb = R.transpose() * (z - p);
  const auto zb = R.transpose() * z;
  const auto v = x.segment<3>(3);
  const auto R_dot = dx.tail<9>().reshaped(3, 3);
  // const auto zb_dot = R_dot.transpose() * (z - p) - R.transpose() * v;
  const auto zb_dot = R_dot.transpose() * z;

  const auto grad = 2 * Q_ * zb;

  if (has_flag(flag, DerivativeFlags::f)) {
    const double level =
        coeff_(0) +
        2 / kPi * coeff_(1) * std::atan(coeff_(2) * p(2) + coeff_(3));
    derivatives_.f(0) = zb.transpose() * Q_ * zb - level;
  }
  if (has_flag(flag, DerivativeFlags::f_z) ||
      has_flag(flag, DerivativeFlags::f_x)) {
    const double level_dot = 2 / kPi * coeff_(1) * coeff_(2) * v(2) /
                             (1 + std::pow(coeff_(2) * p(2) + coeff_(3), 2));
    derivatives_.f_z = grad.transpose() * R.transpose();
    derivatives_.f_x(0) = grad.dot(zb_dot) - level_dot;
  }
  if (has_flag(flag, DerivativeFlags::f_zz_y) ||
      has_flag(flag, DerivativeFlags::f_zz_y_lb) ||
      has_flag(flag, DerivativeFlags::f_xz_y)) {
    const auto hess = 2 * Q_;
    derivatives_.f_zz_y = y(0) * R * hess * R.transpose();
    derivatives_.f_zz_y_lb = derivatives_.f_zz_y;  // Hack.
    derivatives_.f_xz_y = y(0) * (R * hess * zb_dot + R_dot * grad);
  }
  return derivatives_;
}

void QuadrotorUncertainty::LieDerivatives(const VectorXd& x, const VectorXd& z,
                                          const VectorXd& y, const MatrixXd& fg,
                                          MatrixXd& L_fg_y) const {
  assert(x.rows() == kNx);
  assert(z.rows() == kNz);
  assert(y.rows() == kNr);
  assert(fg.rows() == kNdx);
  assert(L_fg_y.rows() == 1);
  assert(L_fg_y.cols() == fg.cols());

  const double kPi = static_cast<double>(EIGEN_PI);

  const auto p = x.head<3>();
  const auto R = x.tail<9>().reshaped(3, 3);
  // const auto zb = R.transpose() * (z - p);
  const auto zb = R.transpose() * z;

  const auto grad = 2 * Q_ * zb;

  for (int i = 0; i < fg.cols(); ++i) {
    const auto v = fg.col(i).head<3>();
    const auto R_dot = fg.col(i).tail<9>().reshaped(3, 3);
    // const auto zb_dot = R_dot.transpose() * (z - p) - R.transpose() * v;
    const auto zb_dot = R_dot.transpose() * z;

    const double level_dot = 2 / kPi * coeff_(1) * coeff_(2) * v(2) /
                             (1 + std::pow(coeff_(2) * p(2) + coeff_(3), 2));
    L_fg_y(0, i) = y(0) * (grad.dot(zb_dot) - level_dot);
  }
}

}  // namespace sccbf
