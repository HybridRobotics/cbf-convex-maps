#include "sccbf/geometry/quadrotor_corridor.h"

#include <Eigen/Core>
#include <cassert>
#include <cmath>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

QuadrotorCorridor::QuadrotorCorridor(double stop_time, double orientation_const,
                                     double max_vel, double margin)
    : ConvexSet(kNz, kNr, kNx, kNdx, margin),
      stop_time_(stop_time),
      orientation_const_(orientation_const),
      max_vel_(max_vel) {
  assert(stop_time > 0);
  assert(orientation_const > 0);
  assert(max_vel > 0);

  CheckDimensions();
}

QuadrotorCorridor::~QuadrotorCorridor() {}

const Derivatives& QuadrotorCorridor::UpdateDerivatives(const VectorXd& x,
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
  const auto v = x.segment<3>(3);
  const auto a = dx.segment<3>(3);
  const auto R_dot = dx.tail<9>().reshaped(3, 3);

  const double v_norm = std::sqrt(std::pow(kEps * max_vel_, 2) + v.dot(v));
  const auto nv = v / v_norm;
  const auto nv_dot = (a - nv * nv.dot(a)) / v_norm;

  const double corridor_time =
      (stop_time_ + orientation_const_ * (1 + R.col(2).dot(nv)));
  const auto q = v * corridor_time;
  const double q_max = max_vel_ * (stop_time_ + 2 * orientation_const_);
  const double q_norm = std::sqrt(std::pow(kEps * q_max, 2) + q.dot(q));
  const auto nq = q / q_norm;
  const auto q_dot =
      a * corridor_time +
      v * orientation_const_ * (R.col(2).dot(nv_dot) + R_dot.col(2).dot(nv));
  const auto nq_dot = (q_dot - nq * nq.dot(q_dot)) / q_norm;

  const auto zb = z - p - q / 2;
  const auto zb_dot = -v - q_dot / 2;

  const auto Q = kEcc * kEcc * MatrixXd::Identity(3, 3) -
                 (kEcc * kEcc - 1) * nq * nq.transpose();

  if (has_flag(flag, DerivativeFlags::f)) {
    const double func = zb.transpose() * Q * zb;
    const double level = std::pow(kMin + q_norm / 2, 2);
    derivatives_.f(0) = func - level;
  }
  if (has_flag(flag, DerivativeFlags::f_z)) {
    derivatives_.f_z = 2 * zb.transpose() * Q;
  }
  if (has_flag(flag, DerivativeFlags::f_zz_y) ||
      has_flag(flag, DerivativeFlags::f_zz_y_lb)) {
    derivatives_.f_zz_y = 2 * y(0) * Q;
    derivatives_.f_zz_y_lb = derivatives_.f_zz_y;
  }
  if (has_flag(flag, DerivativeFlags::f_x) ||
      has_flag(flag, DerivativeFlags::f_xz_y)) {
    const auto Q_dot = -(kEcc * kEcc - 1) *
                       (nq * nq_dot.transpose() + nq_dot * nq.transpose());
    const double level_dot = (kMin + q_norm / 2) * nq.dot(q_dot);
    derivatives_.f_x(0) =
        zb.transpose() * (2 * Q * zb_dot + Q_dot * zb) - level_dot;
    derivatives_.f_xz_y = 2 * y(0) * (Q_dot * zb + Q * zb_dot);
  }
  return derivatives_;
}

void QuadrotorCorridor::LieDerivatives(const VectorXd& x, const VectorXd& z,
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
  const auto v = x.segment<3>(3);

  const double q_max = max_vel_ * (stop_time_ + 2 * orientation_const_);

  for (int i = 0; i < fg.cols(); ++i) {
    const auto vi = fg.col(i).head<3>();
    const auto a = fg.col(i).segment<3>(3);
    const auto R_dot = fg.col(i).tail<9>().reshaped(3, 3);

    const double v_norm = std::sqrt(std::pow(kEps * max_vel_, 2) + v.dot(v));
    const auto nv = v / v_norm;
    const auto nv_dot = (a - nv * nv.dot(a)) / v_norm;

    const double corridor_time =
        (stop_time_ + orientation_const_ * (1 + R.col(2).dot(nv)));
    const auto q = v * corridor_time;
    const double q_norm = std::sqrt(std::pow(kEps * q_max, 2) + q.dot(q));
    const auto nq = q / q_norm;
    const auto q_dot =
        a * corridor_time +
        v * orientation_const_ * (R.col(2).dot(nv_dot) + R_dot.col(2).dot(nv));
    const auto nq_dot = (q_dot - nq * nq.dot(q_dot)) / q_norm;

    const auto zb = z - p - q / 2;
    const auto zb_dot = -vi - q_dot / 2;

    const auto Q = kEcc * kEcc * MatrixXd::Identity(3, 3) -
                   (kEcc * kEcc - 1) * nq * nq.transpose();
    const auto Q_dot = -(kEcc * kEcc - 1) *
                       (nq * nq_dot.transpose() + nq_dot * nq.transpose());
    const double level_dot = (kMin + q_norm / 2) * nq.dot(q_dot);

    L_fg_y(0, i) =
        y(0) * (zb.transpose() * (2 * Q * zb_dot + Q_dot * zb) - level_dot);
  }
}

}  // namespace sccbf
