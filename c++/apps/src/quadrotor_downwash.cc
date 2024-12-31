#include "quadrotor_downwash.h"

#include <Eigen/Core>
#include <cassert>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

QuadrotorDownwash::QuadrotorDownwash(const MatrixXd& A, const VectorXd& b,
                                     double level, double margin)
    : ConvexSet(kNz, kNr, kNx, kNdx, margin), A_(A), b_(b), level_(level) {
  assert(A.rows() > 0);
  assert(A.rows() == b.rows());
  assert(A.cols() == kNz);

  CheckDimensions();
}

QuadrotorDownwash::~QuadrotorDownwash() {}

const Derivatives& QuadrotorDownwash::UpdateDerivatives(const VectorXd& x,
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

  VectorXd softmax(A_.rows());
  const double lse = LogSumExp(A_ * zb - b_, softmax);
  const auto grad = A_.transpose() * softmax;

  if (has_flag(flag, DerivativeFlags::f)) {
    derivatives_.f(0) = lse - level_;
  }
  if (has_flag(flag, DerivativeFlags::f_z) ||
      has_flag(flag, DerivativeFlags::f_x)) {
    derivatives_.f_z = grad.transpose() * R.transpose();
    derivatives_.f_x = grad.transpose() * zb_dot;
  }
  if (has_flag(flag, DerivativeFlags::f_zz_y) ||
      has_flag(flag, DerivativeFlags::f_zz_y_lb) ||
      has_flag(flag, DerivativeFlags::f_xz_y)) {
    const MatrixXd diag = softmax.asDiagonal();
    const auto hess =
        A_.transpose() * (diag - softmax * softmax.transpose()) * A_;
    derivatives_.f_zz_y = y(0) * R * hess * R.transpose();
    derivatives_.f_zz_y_lb = derivatives_.f_zz_y;  // Hack.
    derivatives_.f_xz_y = y(0) * (R * hess * zb_dot + R_dot * grad);
  }
  return derivatives_;
}

void QuadrotorDownwash::LieDerivatives(const VectorXd& x, const VectorXd& z,
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

  VectorXd softmax(A_.rows());
  const double lse = LogSumExp(A_ * zb - b_, softmax);
  const auto grad = A_.transpose() * softmax;

  for (int i = 0; i < fg.cols(); ++i) {
    const auto v = fg.col(i).head<3>();
    const auto R_dot = fg.col(i).tail<9>().reshaped(3, 3);
    const auto zb_dot = R_dot.transpose() * (z - p) - R.transpose() * v;

    L_fg_y(0, i) = y(0) * grad.transpose() * zb_dot;
  }
}

}  // namespace sccbf
