#include "sccbf/lemke.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "sccbf/data_types.h"

namespace {

using namespace sccbf;

Eigen::IOFormat kVecFmt(4, Eigen::DontAlignCols, ", ", "\n", "[", "]");

testing::AssertionResult AssertComponentwiseNonNegative(
    const char* vec_expr, const char* /*tol_expr*/, const VectorXd& vec,
    double tol) {
  if ((vec.array() >= -tol).all()) return testing::AssertionSuccess();

  return testing::AssertionFailure()
         << vec_expr << " =" << std::endl
         << vec.transpose().format(kVecFmt) << std::endl
         << "is not nonnegative";
}

testing::AssertionResult AssertMutuallyOrthogonal(
    const char* vec1_expr, const char* vec2_expr, const char* /*tol_expr*/,
    const VectorXd& vec1, const VectorXd& vec2, double tol) {
  const double inner_product = vec1.transpose() * vec2;
  if (inner_product < tol) return testing::AssertionSuccess();

  return testing::AssertionFailure()
         << vec1_expr << " =" << std::endl
         << vec1.transpose().format(kVecFmt) << std::endl
         << "and " << vec2_expr << " = " << std::endl
         << vec2.transpose().format(kVecFmt) << std::endl
         << "are not orthogonal";
}

testing::AssertionResult AssertVectorEQ(const char* vec1_expr,
                                        const char* vec2_expr,
                                        const char* /*tol_expr*/,
                                        const VectorXd& vec1,
                                        const VectorXd& vec2, double tol) {
  const double inf_norm = (vec1 - vec2).lpNorm<Eigen::Infinity>();
  if (inf_norm < tol) return testing::AssertionSuccess();

  return testing::AssertionFailure()
         << vec1_expr << " =" << std::endl
         << vec1.transpose().format(kVecFmt) << std::endl
         << "and " << vec2_expr << " = " << std::endl
         << vec2.transpose().format(kVecFmt) << std::endl
         << "are not equal";
}

TEST(LemkeTest, OneDimensionInfeasible) {
  MatrixXd M(1, 1);
  M << 0;
  VectorXd q(1);
  q << -1;
  VectorXd z(1);

  auto status = SolveLcp(M, q, z);
  ASSERT_EQ(status, LcpStatus::kInfeasible)
      << "Infeasible LCP returned a solution";
}

TEST(LemkeTest, OneDimensionOptimal) {
  MatrixXd M(1, 1);
  M << 2;
  VectorXd q(1);
  q << -1;
  VectorXd z(1);
  VectorXd sol(1);
  sol << 0.5;

  auto status = SolveLcp(M, q, z);
  ASSERT_EQ(status, LcpStatus::kOptimal) << "LCP not solved to optimality";

  VectorXd w = M * z + q;

  EXPECT_PRED_FORMAT3(AssertMutuallyOrthogonal, z, w, 1e-6);
  EXPECT_PRED_FORMAT2(AssertComponentwiseNonNegative, z, 1e-9);
  EXPECT_PRED_FORMAT2(AssertComponentwiseNonNegative, w, 1e-6);
  EXPECT_PRED_FORMAT3(AssertVectorEQ, z, sol, 1e-6);
}

// Example 2.2 in reference.
TEST(LemkeTest, TwoDimensionOptimal) {
  MatrixXd M(2, 2);
  M << -2, 1, 1, -2;
  VectorXd q(2);
  q << -1, -1;
  VectorXd z(2);

  auto status = SolveLcp(M, q, z);
  ASSERT_EQ(status, LcpStatus::kInfeasible)
      << "Infeasible LCP returned a solution";
}

// Example 2.9 in reference.
TEST(LemkeTest, ThreeDimensionInfeasible) {
  MatrixXd M(3, 3);
  M << -1, 0, -3, 1, -2, -5, -2, -1, -2;
  VectorXd q(3);
  q << -3, -2, -1;
  VectorXd z(3);

  auto status = SolveLcp(M, q, z);
  ASSERT_EQ(status, LcpStatus::kInfeasible)
      << "Infeasible LCP returned solution";
}

// Example 2.10 in reference.
TEST(LemkeTest, ThreeDimensionOptimal) {
  MatrixXd M(3, 3);
  M << 1, 0, 0, 2, 1, 0, 2, 2, 1;
  VectorXd q(3);
  q << -8, -12, -14;
  VectorXd z(3);
  VectorXd sol(3);
  sol << 8, 0, 0;

  auto status = SolveLcp(M, q, z);
  ASSERT_EQ(status, LcpStatus::kOptimal) << "LCP not solved to optimality";

  VectorXd w = M * z + q;
  EXPECT_PRED_FORMAT3(AssertMutuallyOrthogonal, z, w, 1e-6);
  EXPECT_PRED_FORMAT2(AssertComponentwiseNonNegative, z, 1e-9);
  EXPECT_PRED_FORMAT2(AssertComponentwiseNonNegative, w, 1e-6);
  EXPECT_PRED_FORMAT3(AssertVectorEQ, z, sol, 1e-6)
      << "Not the expected solution from Lemke's algorithm (solution might be "
         "correct)";
}

// Example 2.8 in reference.
TEST(LemkeTest, FourDimensionOptimal) {
  MatrixXd M(4, 4);
  M << 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 2, 0, 1, 1, 0, 2;
  VectorXd q(4);
  q << 3, 5, -9, -5;
  VectorXd z(4);
  VectorXd sol(4);
  sol << 2, 1, 3, 1;

  auto status = SolveLcp(M, q, z);
  ASSERT_EQ(status, LcpStatus::kOptimal) << "LCP not solved to optimality";

  VectorXd w = M * z + q;
  EXPECT_PRED_FORMAT3(AssertMutuallyOrthogonal, z, w, 1e-6);
  EXPECT_PRED_FORMAT2(AssertComponentwiseNonNegative, z, 1e-9);
  EXPECT_PRED_FORMAT2(AssertComponentwiseNonNegative, w, 1e-6);
  EXPECT_PRED_FORMAT3(AssertVectorEQ, z, sol, 1e-6)
      << "Not the expected solution from Lemke's algorithm (solution might be "
         "correct)";
}

// Cycling example by M.M. Kostreva (1979).
TEST(LemkeTest, CyclingExample) {
  MatrixXd M(3, 3);
  M << 1, 2, 0, 0, 1, 2, 2, 0, 1;
  VectorXd q(3);
  q << -1, -1, -1;
  VectorXd z(3);
  VectorXd sol(3);
  sol << 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0;

  auto status = SolveLcp(M, q, z);
  ASSERT_EQ(status, LcpStatus::kOptimal) << "LCP not solved to optimality";

  VectorXd w = M * z + q;
  EXPECT_PRED_FORMAT3(AssertMutuallyOrthogonal, z, w, 1e-6);
  EXPECT_PRED_FORMAT2(AssertComponentwiseNonNegative, z, 1e-9);
  EXPECT_PRED_FORMAT2(AssertComponentwiseNonNegative, w, 1e-6);
  EXPECT_PRED_FORMAT3(AssertVectorEQ, z, sol, 1e-6)
      << "Not the expected solution from Lexico Lemke's algorithm "
         "(solution might be correct)";
}

}  // namespace
