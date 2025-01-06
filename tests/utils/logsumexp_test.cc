#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <random>

#include "sccbf/data_types.h"
#include "sccbf/utils/matrix_utils.h"

namespace {

using namespace sccbf;

Eigen::IOFormat kVecFmt(4, Eigen::DontAlignCols, ", ", "\n", "[", "]");

testing::AssertionResult AssertComponentwiseNonNegative(const char* vec_expr,
                                                        const VectorXd& vec) {
  if ((vec.array() >= 0).all()) return testing::AssertionSuccess();

  return testing::AssertionFailure()
         << vec_expr << " =" << std::endl
         << vec.transpose().format(kVecFmt) << std::endl
         << "is not nonnegative";
}

testing::AssertionResult AssertComponentwiseFinite(const char* vec_expr,
                                                   const VectorXd& vec) {
  bool finite = true;
  for (int i = 0; i < vec.rows(); ++i) {
    finite = finite & std::isfinite(vec(i));
  }
  if (finite) return testing::AssertionSuccess();

  return testing::AssertionFailure()
         << vec_expr << " =" << std::endl
         << vec.transpose().format(kVecFmt) << std::endl
         << "is not finite";
}

// LogSumExp tests are taken from the reference.
TEST(LogSumExpTest, ZeroVector) {
  const int n = 100;
  VectorXd vec(n), softmax(n);
  vec = VectorXd::Zero(n);

  const double lse = LogSumExp(vec, softmax);

  VectorXd err(n);
  err.array() = softmax.array() - 1.0 / n;
  EXPECT_PRED_FORMAT1(AssertComponentwiseNonNegative, softmax);
  EXPECT_DOUBLE_EQ(lse, std::log(n));
  EXPECT_DOUBLE_EQ(err.lpNorm<Eigen::Infinity>(), 0);
}

TEST(LogSumExpTest, Overflow) {
  VectorXd vec(3), softmax(3);
  vec << 10000, 1, 1;

  const double lse = LogSumExp(vec, softmax);
  EXPECT_PRED_FORMAT1(AssertComponentwiseNonNegative, softmax);
  EXPECT_TRUE(std::isfinite(lse)) << "Overflow in LSE computation";
  EXPECT_PRED_FORMAT1(AssertComponentwiseFinite, softmax)
      << "Overflow in softmax computation";
  EXPECT_DOUBLE_EQ(softmax.sum(), 1);
}

TEST(LogSumExpTest, Underflow) {
  VectorXd vec(3), softmax(3);
  vec << -10000, -10000, 1;

  VectorXd sol(3);
  sol << 0, 0, 1;
  const double lse = LogSumExp(vec, softmax);
  EXPECT_PRED_FORMAT1(AssertComponentwiseNonNegative, softmax);
  EXPECT_DOUBLE_EQ((softmax - sol).lpNorm<Eigen::Infinity>(), 0);
  EXPECT_DOUBLE_EQ(lse, vec(2));
}

TEST(LogSumExpTest, SmallRandomVector) {
  // Set random generator.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-1, 1);

  const int n = 100;
  VectorXd vec(n), softmax(n);
  vec = VectorXd::NullaryExpr(n, [&dis, &gen]() { return dis(gen); });

  const double lse = LogSumExp(vec, softmax);

  VectorXd err(n);
  err.array() = softmax.array() - vec.array().exp() / vec.array().exp().sum();
  EXPECT_PRED_FORMAT1(AssertComponentwiseNonNegative, softmax);
  EXPECT_DOUBLE_EQ(lse, std::log(vec.array().exp().sum()));
  EXPECT_NEAR(err.lpNorm<Eigen::Infinity>(), 0, 1e-10);
}

TEST(LogSumExpTest, LargeRandomVector) {
  // Set random generator.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dis(0, 100);

  const int n = 100;
  VectorXd vec(n), softmax(n);
  vec = VectorXd::NullaryExpr(n, [&dis, &gen]() { return dis(gen); });

  LogSumExp(vec, softmax);

  EXPECT_PRED_FORMAT1(AssertComponentwiseNonNegative, softmax);
  EXPECT_DOUBLE_EQ(softmax.sum(), 1);
}

}  // namespace
