#include "sccbf/lemke.h"

#include <gtest/gtest.h>

#include "sccbf/data_types.h"

namespace {

using namespace sccbf;

bool MutuallyOrthogonal(const VectorXd& vec1, const VectorXd& vec2,
                        double tol) {
  return (vec1.transpose() * vec2 < tol);
}

bool ComponentwiseNonNegative(const VectorXd& vec1, double tol) {
  return (vec1.array() >= -tol).all();
}

bool VectorEQ(const VectorXd& vec1, const VectorXd& vec2, double tol) {
  return ((vec1 - vec2).norm() <= tol);
}

TEST(LemkeTest, OneDimensionInfeasible) {
  MatrixXd M(1, 1);
  M << 0;
  VectorXd q(1);
  q << -1;
  VectorXd z(1);

  auto status = solve_LCP(M, q, z);
  ASSERT_EQ(status, LCPStatus::INFEASIBLE)
      << "LCP returned solution when infeasible";
}

TEST(LemkeTest, OneDimensionOptimal) {
  MatrixXd M(1, 1);
  M << 2;
  VectorXd q(1);
  q << -1;
  VectorXd z(1);
  VectorXd sol(1);
  sol << 0.5;

  auto status = solve_LCP(M, q, z);
  ASSERT_EQ(status, LCPStatus::OPTIMAL) << "LCP not solved to optimality";

  VectorXd w = M * z + q;
  EXPECT_PRED3(MutuallyOrthogonal, z, w, 1e-6);
  EXPECT_PRED2(ComponentwiseNonNegative, z, 1e-9);
  EXPECT_PRED2(ComponentwiseNonNegative, w, 1e-6);
  EXPECT_PRED3(VectorEQ, z, sol, 1e-6);
}

// Example 2.2 in reference.
TEST(LemkeTest, TwoDimensionOptimal) {
  MatrixXd M(2, 2);
  M << -2, 1, 1, -2;
  VectorXd q(2);
  q << -1, -1;
  VectorXd z(2);

  auto status = solve_LCP(M, q, z);
  ASSERT_EQ(status, LCPStatus::INFEASIBLE)
      << "LCP returned solution when infeasible";
}

// Example 2.9 in reference.
TEST(LemkeTest, ThreeDimensionInfeasible) {
  MatrixXd M(3, 3);
  M << -1, 0, -3, 1, -2, -5, -2, -1, -2;
  VectorXd q(3);
  q << -3, -2, -1;
  VectorXd z(3);

  auto status = solve_LCP(M, q, z);
  ASSERT_EQ(status, LCPStatus::INFEASIBLE)
      << "LCP returned solution when infeasible";
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

  auto status = solve_LCP(M, q, z);
  ASSERT_EQ(status, LCPStatus::OPTIMAL) << "LCP not solved to optimality";

  VectorXd w = M * z + q;
  EXPECT_PRED3(MutuallyOrthogonal, z, w, 1e-6);
  EXPECT_PRED2(ComponentwiseNonNegative, z, 1e-9);
  EXPECT_PRED2(ComponentwiseNonNegative, w, 1e-6);
  EXPECT_PRED3(VectorEQ, z, sol, 1e-6)
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

  auto status = solve_LCP(M, q, z);
  ASSERT_EQ(status, LCPStatus::OPTIMAL) << "LCP not solved to optimality";

  VectorXd w = M * z + q;
  EXPECT_PRED3(MutuallyOrthogonal, z, w, 1e-6);
  EXPECT_PRED2(ComponentwiseNonNegative, z, 1e-9);
  EXPECT_PRED2(ComponentwiseNonNegative, w, 1e-6);
  EXPECT_PRED3(VectorEQ, z, sol, 1e-6)
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

  auto status = solve_LCP(M, q, z);
  ASSERT_EQ(status, LCPStatus::OPTIMAL) << "LCP not solved to optimality";

  VectorXd w = M * z + q;
  EXPECT_PRED3(MutuallyOrthogonal, z, w, 1e-6);
  EXPECT_PRED2(ComponentwiseNonNegative, z, 1e-9);
  EXPECT_PRED2(ComponentwiseNonNegative, w, 1e-6);
  EXPECT_PRED3(VectorEQ, z, sol, 1e-6)
      << "Not the expected solution from Lexico Lemke's algorithm "
         "(solution might be correct)";
}

}  // namespace
