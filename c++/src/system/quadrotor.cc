#include "sccbf/system/quadrotor.h"

#include <Eigen/Core>
#include <cassert>
#include <stdexcept>

#include "sccbf/data_types.h"
#include "sccbf/system/dynamical_system.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

Quadrotor::Quadrotor(double mass, const MatrixXd& inertia,
                     const MatrixXd& constr_mat_u, const VectorXd& constr_vec_u)
    : DynamicalSystem(kNx, kNu, constr_mat_u, constr_vec_u),
      nru_(static_cast<int>(constr_mat_u.rows())),
      mass_(mass),
      inertia_(inertia),
      inertia_inv_(3, 3) {
  assert(mass > 0);
  assert((inertia.rows() == 3) && (inertia.cols() == 3));
  if (!IsPositiveDefinite(inertia)) {
    throw std::runtime_error("Inertia matrix is not positive definite!");
  }
  inertia_inv_ = inertia.inverse();

  CheckDimensions();
}

Quadrotor::~Quadrotor() {}

void Quadrotor::Dynamics(const VectorXd& x, VectorXd& f, MatrixXd& g) const {
  assert(x.rows() == kNx);
  assert(f.rows() == kNx);
  assert((g.rows() == kNx) && (g.cols() == kNu));

  const auto v = x.segment<3>(3);
  const auto R = x.segment<9>(6).reshaped(3, 3);
  const auto w = x.tail<3>();
  MatrixXd w_hat = MatrixXd::Zero(3, 3);
  HatMap<3>(w, w_hat);
  const Eigen::Vector3d gravity(0.0, 0.0, -kGravity);

  f = VectorXd::Zero(kNx);
  f.head<3>() = v;
  f.segment<3>(3) = gravity;
  f.segment<9>(6) = (R * w_hat).reshaped(9, 1);
  f.tail<3>() = -inertia_inv_ * w_hat * inertia_ * w;

  g = MatrixXd::Zero(kNx, kNu);
  g.block<3, 1>(3, 0) = R.col(2) / mass_;
  g.bottomRightCorner<3, 3>() = inertia_inv_;
}

const VectorXd& Quadrotor::IntegrateDynamics(const VectorXd& u, double dt) {
  assert(u.rows() == 4);

  const auto p = x_.head<3>();
  const auto v = x_.segment<3>(3);
  const auto R = x_.segment<9>(6).reshaped(3, 3);
  const auto w = x_.tail<3>();
  MatrixXd w_hat = MatrixXd::Zero(3, 3);
  HatMap<3>(w, w_hat);
  const Eigen::Vector3d gravity(0.0, 0.0, -kGravity);

  const double F = u(0);
  const auto acc = F * R.col(2) / mass_ + gravity;
  const auto M = u.tail<3>();
  const auto ang_acc = inertia_inv_ * (M - w_hat * inertia_ * w);

  x_.head<3>() = p + v * dt + acc * dt * dt / 2;
  x_.segment<3>(3) = v + acc * dt;
  MatrixXd R_new = MatrixXd::Zero(3, 3);
  IntegrateSo3(R, (w + ang_acc * dt / 2) * dt, R_new);
  x_.segment<9>(6) = R_new.reshaped(9, 1);
  x_.tail<3>() = w + ang_acc * dt;
  return x_;
}

}  // namespace sccbf
