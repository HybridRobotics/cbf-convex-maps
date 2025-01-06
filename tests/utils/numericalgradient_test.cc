#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>

#include "sccbf/data_types.h"
#include "sccbf/utils/numerical_derivatives.h"

namespace {

using namespace sccbf;

TEST(NumericalGradientTest, ScalarExponential) {
  const double k = 5;
  const auto func = [k](const VectorXd& x) { return std::exp(k * x(0)); };
  VectorXd x(6);
  x << -1, -0.5, 0, 0.5, 1, 1.5;
  VectorXd sol(x.rows());
  sol.array() = (k * x).array().exp() * k;

  for (int i = 0; i < x.rows(); ++i) {
    VectorXd grad_(1);
    VectorXd xi(1);
    xi << x(i);
    NumericalGradient(func, xi, grad_);
    EXPECT_NEAR(grad_(0), sol(i), 1e-4)
        << "Incorrect derivative at x = " << x(i);
  }
}

TEST(NumericalGradientTest, ScalarLogarithm) {
  const double k = 5;
  const auto func = [k](const VectorXd& x) { return std::log(k * x(0)); };
  VectorXd x(6);
  x << 0.05, 0.1, 1, 10, 100, 1000;
  VectorXd sol(x.rows());
  sol = x.cwiseInverse();

  for (int i = 0; i < x.rows(); ++i) {
    VectorXd grad_(1);
    VectorXd xi(1);
    xi << x(i);
    NumericalGradient(func, xi, grad_);
    EXPECT_NEAR(grad_(0), sol(i), 1e-4)
        << "Incorrect derivative at x = " << x(i);
  }
}

TEST(NumericalGradientTest, VectorSinusoid) {
  const double k1 = 5;
  const double k2 = 10;
  const double kPi = static_cast<double>(EIGEN_PI);
  const auto func = [k1, k2](const VectorXd& x) {
    return std::sin(k1 * x(0)) * std::cos(k2 * x(1));
  };
  VectorXd x1(5);
  x1 << 0, kPi / 6, kPi / 4, kPi / 3, kPi / 2;
  VectorXd x2(x1.rows());
  x2 = x1;

  for (int i = 0; i < x1.rows(); ++i) {
    for (int j = 0; j < x2.rows(); ++j) {
      VectorXd grad_(2);
      VectorXd xi(2);
      xi << x1(i), x2(j);
      NumericalGradient(func, xi, grad_);
      VectorXd sol(2);
      sol(0) = k1 * std::cos(k1 * xi(0)) * std::cos(k2 * xi(1));
      sol(1) = -k2 * std::sin(k1 * xi(0)) * std::sin(k2 * xi(1));
      EXPECT_NEAR((sol - grad_).lpNorm<Eigen::Infinity>(), 0, 1e-4)
          << "Incorrect gradient at x = (" << x1(i) << ", " << x2(i) << ")";
    }
  }
}

}  // namespace
