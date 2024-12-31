#include "quadrotor_shape.h"

#include <Eigen/Core>
#include <cassert>
#include <cmath>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

QuadrotorShape::QuadrotorShape(double pow, const Eigen::Vector4d& coeff,
                               double margin)
    : ConvexSet(kNz, kNr, kNx, kNdx, margin), pow_(pow), coeff_(coeff) {
  assert(pow >= 2);
  assert((coeff.array() > 0).all());

  coeff_pow_.head<3>() = (1.0 / coeff_.head<3>().array()).pow(pow_);
  coeff_pow_(3) = std::pow(coeff_(3), pow_);

  CheckDimensions();
}

QuadrotorShape::~QuadrotorShape() {}

const Derivatives& QuadrotorShape::UpdateDerivatives(const VectorXd& x,
                                                     const VectorXd& dx,
                                                     const VectorXd& z,
                                                     const VectorXd& y,
                                                     DerivativeFlags flag) {
  assert(x.rows() == kNx);
  assert(dx.rows() == kNdx);
  assert(z.rows() == kNz);
  assert(y.rows() == kNr);

  const auto p = x.head<3>();
  const auto R = x.tail<9>().reshaped(3, 3);
  const auto zb = R.transpose() * (z - p);
  const auto v = x.segment<3>(3);
  const auto R_dot = dx.tail<9>().reshaped(3, 3);
  const auto zb_dot = R_dot.transpose() * (z - p) - R.transpose() * v;

  const VectorXd zb_pow2 = zb.cwiseAbs().cwisePow(pow_ - 2);
  const VectorXd grad =
      pow_ * coeff_pow_.head<3>().cwiseProduct(zb_pow2.cwiseProduct(zb));

  if (has_flag(flag, DerivativeFlags::f)) {
    const auto zb_pow = zb_pow2.cwiseProduct(zb.cwiseSquare());
    derivatives_.f(0) = coeff_pow_.head<3>().dot(zb_pow) - coeff_pow_(3);
  }
  if (has_flag(flag, DerivativeFlags::f_z) ||
      has_flag(flag, DerivativeFlags::f_x)) {
    derivatives_.f_z = grad.transpose() * R.transpose();
    derivatives_.f_x = grad.transpose() * zb_dot;
  }
  if (has_flag(flag, DerivativeFlags::f_zz_y) ||
      has_flag(flag, DerivativeFlags::f_zz_y_lb) ||
      has_flag(flag, DerivativeFlags::f_xz_y)) {
    const auto hess = pow_ * (pow_ - 1) *
                      coeff_pow_.head<3>().cwiseProduct(zb_pow2).asDiagonal();
    derivatives_.f_zz_y = y(0) * R * hess * R.transpose();
    derivatives_.f_zz_y_lb = derivatives_.f_zz_y;  // Hack.
    derivatives_.f_xz_y = y(0) * (R * hess * zb_dot + R_dot * grad);
  }
  return derivatives_;
}

void QuadrotorShape::LieDerivatives(const VectorXd& x, const VectorXd& z,
                                    const VectorXd& y, const MatrixXd& fg,
                                    MatrixXd& L_fg_y) const {
  assert(x.rows() == kNx);
  assert(z.rows() == kNz);
  assert(y.rows() == kNr);
  assert(fg.rows() == kNdx);
  assert(L_fg_y.rows() == 1);
  assert(L_fg_y.cols() == fg.cols());

  const auto p = x.head<3>();
  const auto R = x.tail<9>().reshaped(3, 3);
  const auto zb = R.transpose() * (z - p);
  const auto zb_pow2 = zb.cwiseAbs().cwisePow(pow_ - 2);
  const auto grad =
      pow_ * coeff_pow_.head<3>().cwiseProduct(zb_pow2.cwiseProduct(zb));

  for (int i = 0; i < fg.cols(); ++i) {
    const auto v = fg.col(i).head<3>();
    const auto R_dot = fg.col(i).tail<9>().reshaped(3, 3);
    const auto zb_dot = R_dot.transpose() * (z - p) - R.transpose() * v;

    L_fg_y(0, i) = y(0) * grad.transpose() * zb_dot;
  }
}

}  // namespace sccbf
