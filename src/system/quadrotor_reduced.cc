#include "sccbf/system/quadrotor_reduced.h"

#include <Eigen/Core>
#include <cassert>
#include <stdexcept>

#include "sccbf/data_types.h"
#include "sccbf/system/dynamical_system.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

QuadrotorReduced::QuadrotorReduced(double mass, const MatrixXd& constr_mat_u,
                                   const VectorXd& constr_vec_u)
    : DynamicalSystem(kNx, kNu, constr_mat_u, constr_vec_u),
      nru_(static_cast<int>(constr_mat_u.rows())),
      mass_(mass) {
  assert(mass > 0);

  CheckDimensions();
}

QuadrotorReduced::~QuadrotorReduced() {}

void QuadrotorReduced::Dynamics(const VectorXd& x, VectorXd& f,
                                MatrixXd& g) const {
  assert(x.rows() == kNx);
  assert(f.rows() == kNx);
  assert((g.rows() == kNx) && (g.cols() == kNu));

  const auto v = x.segment<3>(3);
  const auto R = x.tail<9>().reshaped(3, 3);
  const Eigen::Vector3d gravity(0.0, 0.0, -kGravity);

  f = VectorXd::Zero(kNx);
  f.head<3>() = v;
  f.segment<3>(3) = gravity;

  g = MatrixXd::Zero(kNx, kNu);
  g.block<3, 1>(3, 0) = R.col(2) / mass_;
  g.block<3, 1>(6, 2) = -R.col(2);
  g.block<3, 1>(6, 3) = R.col(1);
  g.block<3, 1>(9, 1) = R.col(2);
  g.block<3, 1>(9, 3) = -R.col(0);
  g.block<3, 1>(12, 1) = -R.col(1);
  g.block<3, 1>(12, 2) = R.col(0);
}

const VectorXd& QuadrotorReduced::IntegrateDynamics(const VectorXd& u,
                                                    double dt) {
  assert(u.rows() == 4);

  const auto p = x_.head<3>();
  const auto v = x_.segment<3>(3);
  const auto R = x_.tail<9>().reshaped(3, 3);
  const Eigen::Vector3d gravity(0.0, 0.0, -kGravity);

  const double F = u(0);
  const auto acc = F * R.col(2) / mass_ + gravity;
  const auto w = u.tail<3>();

  x_.head<3>() = p + v * dt + acc * dt * dt / 2;
  x_.segment<3>(3) = v + acc * dt;
  MatrixXd R_new = MatrixXd::Zero(3, 3);
  IntegrateSo3(R, w * dt, R_new);
  x_.tail<9>() = R_new.reshaped(9, 1);
  return x_;
}

}  // namespace sccbf
