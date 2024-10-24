#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <random>

#include "sccbf/data_types.h"
#include "sccbf/math_utils/utils.h"

namespace {

using namespace sccbf;

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

}  // namespace
