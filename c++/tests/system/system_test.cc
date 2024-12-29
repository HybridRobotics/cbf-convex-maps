#include <gtest/gtest.h>

#include <Eigen/Core>
#include <memory>

#include "sccbf/data_types.h"
#include "sccbf/utils/numerical_derivatives.h"
#include "sccbf/utils/matrix_utils.h"

#include "sccbf/system/dynamical_system.h"
#include "sccbf/system/integrator.h"
#include "sccbf/system/integrator_se.h"
#include "sccbf/system/double_integrator_se3.h"
#include "sccbf/system/unicycle.h"
#include "sccbf/system/unicycle_se2.h"
#include "sccbf/system/quadrotor.h"
#include "sccbf/system/quadrotor_reduced.h"


namespace {

using namespace sccbf;

struct DynamicsVectorField {
  VectorXd f;
  MatrixXd g;

  DynamicsVectorField(int nx, int nu): f(nx), g(nx, nu) {}
};

// Assertion function
Eigen::IOFormat kVecFmt(4, Eigen::DontAlignCols, ", ", "\n", "[", "]");
Eigen::IOFormat kMatFmt(4, 0, ", ", "\n", "[", "]");

testing::AssertionResult AssertDynamicsEQ(
    const char* vf1_expr, const char* vf2_expr, const char* /*x_expr*/,
    const char* /*tol_expr*/, const DynamicsVectorField& vf1, const DynamicsVectorField& vf2,
    const VectorXd& x, double tol) {
  auto failure = testing::AssertionFailure();

  bool success = true;

  if ((vf1.f - vf2.f).lpNorm<Eigen::Infinity>() >= tol) {
    success = false;
    failure << vf1_expr << ".f = " << std::endl
            << vf1.f.transpose().format(kVecFmt) << std::endl
            << "is not equal to " << vf2_expr << ".f = " << std::endl
            << vf2.f.transpose().format(kVecFmt) << std::endl;
  }
  if ((vf1.g - vf2.g).lpNorm<Eigen::Infinity>() >= tol) {
    success = false;
    failure << vf1_expr << ".g = " << std::endl
            << vf1.g.format(kMatFmt) << std::endl
            << "is not equal to " << vf2_expr << ".g = " << std::endl
            << vf2.g.format(kMatFmt) << std::endl;
  }

  if (success) return testing::AssertionSuccess();

  failure << "Failure at x = " << std::endl
          << x.transpose().format(kVecFmt);

  return failure;
}

TEST(DynamicalSystemTest, Integrator3d) {
  const double dt = 1000 * 1e-6;
  const int n = 3;
  const int nx = n;
  const int nu = n;
  auto sys = Integrator3d(MatrixXd::Zero(0, nu), VectorXd::Zero(0));
  sys.set_x(VectorXd::Random(nx));

  const VectorXd u = VectorXd::Random(n);
  const VectorXd& x = sys.x();
  const VectorXd x_new = x + u * dt;

  EXPECT_TRUE(x_new.isApprox(sys.IntegrateDynamics(u, dt), 1e-4));

  DynamicsVectorField vf(nx, nu), vf_numerical(nx, nu);
  const VectorXd x_copy = sys.x();
  sys.Dynamics(x_copy, vf.f, vf.g);
  NumericalDynamics(sys, x_copy, vf_numerical.f, vf_numerical.g);

  EXPECT_PRED_FORMAT4(AssertDynamicsEQ, vf, vf_numerical, x_copy, 1e-3);
}

TEST(DynamicalSystemTest, IntegratorSe2) {
  const double dt = 1000 * 1e-6;
  const int nx = 2 + 4;
  const int nu = 2 + 1;
  auto sys = IntegratorSe2(MatrixXd::Zero(0, nu), VectorXd::Zero(0));
  VectorXd x(nx);
  x.head<2>() = VectorXd::Random(2);
  MatrixXd R(2, 2);
  RandomRotation<2>(R);
  x.tail<4>() = R.reshaped(4, 1);
  sys.set_x(x);

  const VectorXd u = VectorXd::Random(nu);
  VectorXd x_new(nx);
  x_new.head<2>() = x.head<2>() + u.head<2>() * dt;
  MatrixXd R_new(2, 2);
  IntegrateSo2(R, u(2) * dt, R_new);
  x_new.tail<4>() = R_new.reshaped(4, 1);

  EXPECT_TRUE(x_new.isApprox(sys.IntegrateDynamics(u, dt), 1e-4));

  DynamicsVectorField vf(nx, nu), vf_numerical(nx, nu);
  const VectorXd x_copy = sys.x();
  sys.Dynamics(x_copy, vf.f, vf.g);
  NumericalDynamics(sys, x_copy, vf_numerical.f, vf_numerical.g);

  EXPECT_PRED_FORMAT4(AssertDynamicsEQ, vf, vf_numerical, x_copy, 1e-3);
}

TEST(DynamicalSystemTest, IntegratorSe3) {
  const double dt = 1000 * 1e-6;
  const int nx = 3 + 9;
  const int nu = 3 + 3;
  auto sys = IntegratorSe3(MatrixXd::Zero(0, nu), VectorXd::Zero(0));
  VectorXd x(nx);
  x.head<3>() = VectorXd::Random(3);
  MatrixXd R(3, 3);
  RandomRotation<3>(R);
  x.tail<9>() = R.reshaped(9, 1);
  sys.set_x(x);

  const VectorXd u = VectorXd::Random(nu);
  VectorXd x_new(nx);
  x_new.head<3>() = x.head<3>() + u.head<3>() * dt;
  MatrixXd R_new(3, 3);
  IntegrateSo3(R, u.tail<3>() * dt, R_new);
  x_new.tail<9>() = R_new.reshaped(9, 1);

  EXPECT_TRUE(x_new.isApprox(sys.IntegrateDynamics(u, dt), 1e-4));

  DynamicsVectorField vf(nx, nu), vf_numerical(nx, nu);
  const VectorXd x_copy = sys.x();
  sys.Dynamics(x_copy, vf.f, vf.g);
  NumericalDynamics(sys, x_copy, vf_numerical.f, vf_numerical.g);

  EXPECT_PRED_FORMAT4(AssertDynamicsEQ, vf, vf_numerical, x_copy, 1e-3);
}

TEST(DynamicalSystemTest, Unicycle) {
  const double kPi = static_cast<double>(EIGEN_PI);
  const double dt = 1000 * 1e-6;
  const int nx = 2 + 1;
  const int nu = 1 + 1;
  auto sys = Unicycle(MatrixXd::Zero(0, nu), VectorXd::Zero(0));
  VectorXd x(nx);
  x = VectorXd::Random(3);
  x(2) *= kPi;
  sys.set_x(x);

  const VectorXd u = VectorXd::Random(nu);
  VectorXd x_new(nx);
  x_new(0) = x(0) + std::cos(x(2)) * u(0) * dt;
  x_new(1) = x(1) + std::sin(x(2)) * u(0) * dt;
  x_new(2) = x(2) + u(1) * dt;

  EXPECT_TRUE(x_new.isApprox(sys.IntegrateDynamics(u, dt), 1e-4));

  DynamicsVectorField vf(nx, nu), vf_numerical(nx, nu);
  const VectorXd x_copy = sys.x();
  sys.Dynamics(x_copy, vf.f, vf.g);
  NumericalDynamics(sys, x_copy, vf_numerical.f, vf_numerical.g);

  EXPECT_PRED_FORMAT4(AssertDynamicsEQ, vf, vf_numerical, x_copy, 1e-3);
}

TEST(DynamicalSystemTest, UnicycleSe2) {
  const double dt = 1000 * 1e-6;
  const int nx = 2 + 4;
  const int nu = 2;
  auto sys = UnicycleSe2(MatrixXd::Zero(0, nu), VectorXd::Zero(0));
  VectorXd x(nx);
  x.head<2>() = VectorXd::Random(2);
  MatrixXd R(2, 2);
  RandomRotation<2>(R);
  x.tail<4>() = R.reshaped(4, 1);
  sys.set_x(x);

  const VectorXd u = VectorXd::Random(nu);
  VectorXd x_new(nx);
  x_new(0) = x(0) + x(2) * u(0) * dt;
  x_new(1) = x(1) + x(3) * u(0) * dt;
  MatrixXd R_new(2, 2);
  IntegrateSo2(R, u(1) * dt, R_new);
  x_new.tail<4>() = R_new.reshaped(4, 1);

  EXPECT_TRUE(x_new.isApprox(sys.IntegrateDynamics(u, dt), 1e-4));

  DynamicsVectorField vf(nx, nu), vf_numerical(nx, nu);
  const VectorXd x_copy = sys.x();
  sys.Dynamics(x_copy, vf.f, vf.g);
  NumericalDynamics(sys, x_copy, vf_numerical.f, vf_numerical.g);

  EXPECT_PRED_FORMAT4(AssertDynamicsEQ, vf, vf_numerical, x_copy, 1e-3);
}

constexpr double kGravity = 9.81; // [m/s^2].

TEST(DynamicalSystemTest, Quadrotor) {
  const int nx = 3 + 3 + 9 + 3;
  const int nu = 1 + 3;
  const double mass = 0.5; // [kg].
  const Eigen::Vector3d inertia_diag(2.32 * 1e-3, 2.32 * 1e-3, 4 * 1e-3);
  const MatrixXd inertia = inertia_diag.asDiagonal();
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);

  std::shared_ptr<DynamicalSystem> sys = std::make_shared<Quadrotor>(mass, inertia, constr_mat_u, constr_vec_u);
  VectorXd x(nx);
  x.head<6>() = VectorXd::Random(6);
  MatrixXd R(3, 3);
  RandomRotation<3>(R);
  x.segment<9>(6) = R.reshaped(9, 1);
  x.tail<3>() = VectorXd::Random(3);
  sys->set_x(x);

  DynamicsVectorField vf(nx, nu), vf_numerical(nx, nu);
  const VectorXd x_copy = sys->x();
  sys->Dynamics(x_copy, vf.f, vf.g);
  NumericalDynamics(*sys, x_copy, vf_numerical.f, vf_numerical.g);

  EXPECT_PRED_FORMAT4(AssertDynamicsEQ, vf, vf_numerical, x_copy, 1e-3);
}

TEST(DynamicalSystemTest, QuadrotorReduced) {
  const int nx = 3 + 3 + 9;
  const int nu = 1 + 3;
  const double mass = 0.5; // [kg].
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);

  std::shared_ptr<DynamicalSystem> sys = std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u);
  VectorXd x(nx);
  x.head<6>() = VectorXd::Random(6);
  MatrixXd R(3, 3);
  RandomRotation<3>(R);
  x.tail<9>() = R.reshaped(9, 1);
  sys->set_x(x);

  DynamicsVectorField vf(nx, nu), vf_numerical(nx, nu);
  const VectorXd x_copy = sys->x();
  sys->Dynamics(x_copy, vf.f, vf.g);
  NumericalDynamics(*sys, x_copy, vf_numerical.f, vf_numerical.g);

  EXPECT_PRED_FORMAT4(AssertDynamicsEQ, vf, vf_numerical, x_copy, 1e-3);
}

} // namespace