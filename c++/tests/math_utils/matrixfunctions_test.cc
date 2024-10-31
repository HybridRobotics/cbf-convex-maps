#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <random>

#include "sccbf/data_types.h"
#include "sccbf/math_utils/utils.h"

namespace {

using namespace sccbf;

Eigen::IOFormat kMatFmt(4, 0, ", ", "\n", "[", "]");

testing::AssertionResult AssertMatrixEQ(const char* mat1_expr,
                                        const char* mat2_expr,
                                        const char* /*tol_expr*/,
                                        const MatrixXd& mat1,
                                        const MatrixXd& mat2, double tol) {
  const double inf_norm = (mat1 - mat2).lpNorm<Eigen::Infinity>();
  if (inf_norm < tol) return testing::AssertionSuccess();

  return testing::AssertionFailure()
         << mat1_expr << " =" << std::endl
         << mat1.format(kMatFmt) << std::endl
         << "and " << mat2_expr << " = " << std::endl
         << mat2.format(kMatFmt) << std::endl
         << "are not equal";
}

TEST(MatrixFunctionTest, HatMap2) {
  VectorXd vec(1);
  vec << 1;
  MatrixXd hat(2, 2);
  hat = MatrixXd::Zero(2, 2);
  MatrixXd sol(2, 2);
  sol << 0, -1, 1, 0;

  HatMap<2>(vec, hat);
  EXPECT_PRED_FORMAT3(AssertMatrixEQ, hat, sol, 1e-9);
}

TEST(MatrixFunctionTest, HatMap3) {
  VectorXd vec(3);
  vec << 1, 2, 3;
  MatrixXd hat(3, 3);
  hat = MatrixXd::Zero(3, 3);
  MatrixXd sol(3, 3);
  sol << 0, -3, 2, 3, 0, -1, -2, 1, 0;

  HatMap<3>(vec, hat);
  EXPECT_PRED_FORMAT3(AssertMatrixEQ, hat, sol, 1e-9);
}

TEST(MatrixFunctionTest, EulerToRotationMatrix) {
  // Set random generator.
  std::random_device rd;
  std::mt19937 gen(rd());
  const double kPi = static_cast<double>(EIGEN_PI);
  std::uniform_real_distribution<double> dis(-kPi, kPi);

  const double angle_x = dis(gen);
  const double angle_y = dis(gen) / 2.0;
  const double angle_z = dis(gen);
  MatrixXd rotation(3, 3);

  EulerToRotation(angle_x, angle_y, angle_z, rotation);

  EXPECT_TRUE(rotation.isUnitary());
  EXPECT_NEAR(rotation.determinant(), 1, 1e-6);
}

TEST(MatrixFunctionTest, RandomRotation2) {
  MatrixXd rotation(2, 2);
  RandomRotation<2>(rotation);

  EXPECT_TRUE(rotation.isUnitary());
  EXPECT_NEAR(rotation.determinant(), 1, 1e-6);
}

TEST(MatrixFunctionTest, RandomRotation3) {
  MatrixXd rotation(3, 3);
  RandomRotation<3>(rotation);

  EXPECT_TRUE(rotation.isUnitary());
  EXPECT_NEAR(rotation.determinant(), 1, 1e-6);
}

TEST(MatrixFunctionTest, IsPositiveDefinite) {
  MatrixXd rotation(3, 3);
  RandomRotation<3>(rotation);

  {
    VectorXd eig(3);
    eig << -1, 0, 1;
    MatrixXd mat = rotation * eig.asDiagonal() * rotation.transpose();

    EXPECT_FALSE(IsPositiveDefinite(mat));
  }

  {
    VectorXd eig(3);
    eig << 0.01, 0.01, 1;
    MatrixXd mat = rotation * eig.asDiagonal() * rotation.transpose();

    EXPECT_TRUE(IsPositiveDefinite(mat));
  }
}

}  // namespace
