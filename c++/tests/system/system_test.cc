#include <gtest/gtest.h>

#include <Eigen/Core>
#include <memory>

#include "sccbf/data_types.h"
#include "sccbf/system/dynamical_system.h"
#include "sccbf/system/integrator.h"

namespace {

using namespace sccbf;

std::shared_ptr<DynamicalSystem> GetIntegrator(int n) {
  assert((n >= 2) && (n <= 4));
  MatrixXd constr_mat_u(2 * n, n);
  constr_mat_u.topRows(n) = MatrixXd::Identity(n, n);
  constr_mat_u.bottomRows(n) = -MatrixXd::Identity(n, n);
  VectorXd constr_vec_u = VectorXd::Ones(2 * n);

  if (n == 2) {
    return std::make_shared<Integrator2d>(constr_mat_u, constr_vec_u);
  }
  else if (n == 3) {
    return std::make_shared<Integrator3d>(constr_mat_u, constr_vec_u);
  }
  else {
    return std::make_shared<Integrator4d>(constr_mat_u, constr_vec_u);
  }
}

TEST(DynamicalSystemTest, Integrator3d) {
  const double dt = 1000 * 1e-6;
  const int n = 3;
  std::shared_ptr<DynamicalSystem> sys = GetIntegrator(n);

  VectorXd u = VectorXd::Random(n);
  const VectorXd& x = sys->x();
  const VectorXd x_new = x + u * dt;

  EXPECT_TRUE(x_new.isApprox(sys->IntegrateDynamics(u, dt), 1e-4));

  // TODO: Numerical derivative, check dx.
}

} // namespace