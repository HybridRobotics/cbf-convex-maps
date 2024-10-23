#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <random>

#include "sccbf/data_types.h"
#include "sccbf/math_utils/numerical_derivatives.h"
#include "sccbf/math_utils/utils.h"

namespace {

using namespace sccbf;

// Assumes sizes of matrices are equal.
void MatrixDoubleEQ(const MatrixXd& mat1, const MatrixXd& mat2) {
  for (int i = 0; i < mat1.rows(); ++i) {
    for (int j = 0; j < mat1.cols(); ++j) {
      EXPECT_DOUBLE_EQ(mat1(i, j), mat2(i, j))
          << "Matrices don't match at index (i, j): (" << i << ", " << j << ")";
    }
  }
}

bool ComponentwiseNonNegative(const VectorXd& vec) {
  return (vec.array() >= 0).all();
}

bool ComponentwiseFinite(const VectorXd& vec) {
  bool finite = true;
  for (int i = 0; i < vec.rows(); ++i) {
    finite = finite & std::isfinite(vec(i));
  }
  return finite;
}

// LogSumExp tests are taken from the reference.
TEST(LogSumExp, ZeroVector) {
  const int n = 100;
  VectorXd vec(n), softmax(n);
  vec = VectorXd::Zero(n);

  const double lse = log_sum_exp(vec, softmax);

  VectorXd err(n);
  err.array() = softmax.array() - 1.0 / n;
  EXPECT_TRUE(ComponentwiseNonNegative(softmax));
  EXPECT_DOUBLE_EQ(lse, std::log(n));
  EXPECT_DOUBLE_EQ(err.lpNorm<Eigen::Infinity>(), 0);
}

TEST(LogSumExp, Overflow) {
  VectorXd vec(3), softmax(3);
  vec << 10000, 1, 1;

  const double lse = log_sum_exp(vec, softmax);
  EXPECT_TRUE(ComponentwiseNonNegative(softmax));
  EXPECT_TRUE(std::isfinite(lse)) << "Overflow in LSE computation";
  EXPECT_TRUE(ComponentwiseFinite(softmax))
      << "Overflow in softmax computation";
  EXPECT_DOUBLE_EQ(softmax.sum(), 1);
}

TEST(LogSumExp, Underflow) {
  VectorXd vec(3), softmax(3);
  vec << -10000, -10000, 1;

  VectorXd sol(3);
  sol << 0, 0, 1;
  const double lse = log_sum_exp(vec, softmax);
  EXPECT_TRUE(ComponentwiseNonNegative(softmax));
  EXPECT_DOUBLE_EQ((softmax - sol).lpNorm<Eigen::Infinity>(), 0);
  EXPECT_DOUBLE_EQ(lse, vec(2));
}

TEST(LogSumExp, SmallRandomVector) {
  // Set random generator.
  std::mt19937 gen(2);
  std::uniform_real_distribution<double> dis(-1, 1);

  const int n = 100;
  VectorXd vec(n), softmax(n);
  vec = VectorXd::NullaryExpr(n, [&dis, &gen]() { return dis(gen); });

  const double lse = log_sum_exp(vec, softmax);

  VectorXd err(n);
  err.array() = softmax.array() - vec.array().exp() / vec.array().exp().sum();
  EXPECT_TRUE(ComponentwiseNonNegative(softmax));
  EXPECT_DOUBLE_EQ(lse, std::log(vec.array().exp().sum()));
  EXPECT_NEAR(err.lpNorm<Eigen::Infinity>(), 0, 1e-10);
}

TEST(LogSumExp, LargeRandomVector) {
  // Set random generator.
  std::mt19937 gen(3);
  std::normal_distribution<double> dis(0, 100);

  const int n = 100;
  VectorXd vec(n), softmax(n);
  vec = VectorXd::NullaryExpr(n, [&dis, &gen]() { return dis(gen); });

  log_sum_exp(vec, softmax);

  EXPECT_TRUE(ComponentwiseNonNegative(softmax));
  EXPECT_DOUBLE_EQ(softmax.sum(), 1);
}

TEST(MatrixTest, HatMap) {
  {
    VectorXd vec(1);
    vec << 1;
    MatrixXd hat(2, 2);
    hat = MatrixXd::Zero(2, 2);
    MatrixXd sol(2, 2);
    sol << 0, -1, 1, 0;

    hat_map<2>(vec, hat);
    MatrixDoubleEQ(hat, sol);
  }

  {
    VectorXd vec(3);
    vec << 1, 2, 3;
    MatrixXd hat(3, 3);
    hat = MatrixXd::Zero(3, 3);
    MatrixXd sol(3, 3);
    sol << 0, -3, 2, 3, 0, -1, -2, 1, 0;

    hat_map<3>(vec, hat);
    MatrixDoubleEQ(hat, sol);
  }
}

TEST(MatrixTest, EulerToRot) {
  // Set random generator.
  std::mt19937 gen(1);
  const double pi = static_cast<double>(EIGEN_PI);
  std::uniform_real_distribution<double> dis(-pi, pi);

  const double ang_x = dis(gen);
  const double ang_y = dis(gen) / 2.0;
  const double ang_z = dis(gen);
  MatrixXd rot(3, 3);

  euler_to_rot(ang_x, ang_y, ang_z, rot);

  EXPECT_TRUE(rot.isUnitary());
  EXPECT_NEAR(rot.determinant(), 1, 1e-6);
}

TEST(MatrixTest, RandomRotation) {
  {
    MatrixXd rot = random_rotation<2>();

    EXPECT_TRUE(rot.isUnitary());
    EXPECT_DOUBLE_EQ(rot.determinant(), 1);
  }

  {
    MatrixXd rot = random_rotation<3>();

    EXPECT_TRUE(rot.isUnitary());
    EXPECT_NEAR(rot.determinant(), 1, 1e-6);
  }
}

TEST(MatrixTest, IsPositiveDefinite) {
  {
    MatrixXd R = random_rotation<3>();
    VectorXd eig(3);
    eig << -1, 0, 1;
    MatrixXd mat = R * eig.asDiagonal() * R.transpose();

    EXPECT_FALSE(is_positive_definite(mat));
  }

  {
    MatrixXd R = random_rotation<3>();
    VectorXd eig(3);
    eig << 0.01, 0.01, 1;
    MatrixXd mat = R * eig.asDiagonal() * R.transpose();

    EXPECT_TRUE(is_positive_definite(mat));
  }
}

TEST(NumericalGradient, ScalarExponential) {
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
    numerical_gradient(func, xi, grad_);
    EXPECT_NEAR(grad_(0), sol(i), 1e-5)
        << "Incorrect derivative at x = " << x(i);
  }
}

TEST(NumericalGradient, ScalarLogarithm) {
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
    numerical_gradient(func, xi, grad_);
    EXPECT_NEAR(grad_(0), sol(i), 1e-5)
        << "Incorrect derivative at x = " << x(i);
  }
}

TEST(NumericalGradient, VectorSinusoid) {
  const double k1 = 5;
  const double k2 = 10;
  const double pi = static_cast<double>(EIGEN_PI);
  const auto func = [k1, k2](const VectorXd& x) {
    return std::sin(k1 * x(0)) * std::cos(k2 * x(1));
  };
  VectorXd x1(5);
  x1 << 0, pi / 6, pi / 4, pi / 3, pi / 2;
  VectorXd x2(x1.rows());
  x2 = x1;

  for (int i = 0; i < x1.rows(); ++i) {
    for (int j = 0; j < x2.rows(); ++j) {
      VectorXd grad_(2);
      VectorXd xi(2);
      xi << x1(i), x2(j);
      numerical_gradient(func, xi, grad_);
      VectorXd sol(2);
      sol(0) = k1 * std::cos(k1 * xi(0)) * std::cos(k2 * xi(1));
      sol(1) = -k2 * std::sin(k1 * xi(0)) * std::sin(k2 * xi(1));
      EXPECT_NEAR((sol - grad_).lpNorm<Eigen::Infinity>(), 0, 1e-5)
          << "Incorrect gradient at x = (" << x1(i) << ", " << x2(i) << ")";
    }
  }
}

}  // namespace
