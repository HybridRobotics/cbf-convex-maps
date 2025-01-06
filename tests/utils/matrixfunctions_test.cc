#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <random>

#include "sccbf/data_types.h"
#include "sccbf/utils/matrix_utils.h"

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

TEST(MatrixFunctionTest, HatMap2d) {
  VectorXd vec(1);
  vec << 1;
  MatrixXd hat(2, 2);
  hat = MatrixXd::Zero(2, 2);
  MatrixXd sol(2, 2);
  sol << 0, -1, 1, 0;

  HatMap<2>(vec, hat);
  EXPECT_PRED_FORMAT3(AssertMatrixEQ, hat, sol, 1e-9);
}

TEST(MatrixFunctionTest, HatMap3d) {
  VectorXd vec(3);
  vec << 1, 2, 3;
  MatrixXd hat(3, 3);
  hat = MatrixXd::Zero(3, 3);
  MatrixXd sol(3, 3);
  sol << 0, -3, 2, 3, 0, -1, -2, 1, 0;

  HatMap<3>(vec, hat);
  EXPECT_PRED_FORMAT3(AssertMatrixEQ, hat, sol, 1e-9);
}

TEST(MatrixFunctionTest, EulerToRotation) {
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

TEST(MatrixFunctionTest, RotationFromZVector) {
  // Generate random z-vector.
  VectorXd z = VectorXd::Random(3);
  z(0) = 2.0 * (z(0) - 0.5);
  z(1) = 2.0 * (z(1) - 0.5);
  z(2) = std::max(z(2), 1e-3);

  MatrixXd rot(3, 3);
  RotationFromZVector(z, rot);
  z = z / z.norm();

  EXPECT_TRUE(rot.isUnitary());
  EXPECT_NEAR(rot.determinant(), 1, 1e-6);
  EXPECT_NEAR((z - rot.col(2)).norm(), 0.0, 1e-6);
}

TEST(MatrixFunctionTest, RandomRotation2d) {
  MatrixXd rotation(2, 2);
  RandomRotation<2>(rotation);

  EXPECT_TRUE(rotation.isUnitary());
  EXPECT_NEAR(rotation.determinant(), 1, 1e-6);
}

TEST(MatrixFunctionTest, RandomRotation3d) {
  MatrixXd rotation(3, 3);
  RandomRotation<3>(rotation);

  EXPECT_TRUE(rotation.isUnitary());
  EXPECT_NEAR(rotation.determinant(), 1, 1e-6);
}

TEST(MatrixFunctionTest, IntegrateSo2) {
  const double kPi = static_cast<double>(EIGEN_PI);
  MatrixXd rot = MatrixXd::Identity(2, 2);

  const double ang1 = kPi / 3;
  MatrixXd sol1(2, 2);
  sol1 << std::cos(ang1), -std::sin(ang1), std::sin(ang1), std::cos(ang1);
  IntegrateSo2(rot, ang1, rot);
  EXPECT_TRUE(rot.isUnitary());
  EXPECT_NEAR(rot.determinant(), 1, 1e-6);
  EXPECT_PRED_FORMAT3(AssertMatrixEQ, rot, sol1, 1e-5);

  const double ang2 = kPi / 9;
  MatrixXd sol2(2, 2);
  sol2 << std::cos(ang2), -std::sin(ang2), std::sin(ang2), std::cos(ang2);
  sol2 = sol1 * sol2;
  IntegrateSo2(rot, ang2, rot);
  EXPECT_TRUE(rot.isUnitary());
  EXPECT_NEAR(rot.determinant(), 1, 1e-6);
  EXPECT_PRED_FORMAT3(AssertMatrixEQ, rot, sol2, 1e-5);
}

TEST(MatrixFunctionTest, IntegrateSo3) {
  const double kPi = static_cast<double>(EIGEN_PI);
  MatrixXd rot = MatrixXd::Identity(3, 3);

  VectorXd yaw(3);
  yaw << 0, 0, kPi / 3;
  MatrixXd sol1(3, 3);
  EulerToRotation(0, 0, yaw(2), sol1);
  IntegrateSo3(rot, yaw, rot);
  EXPECT_TRUE(rot.isUnitary());
  EXPECT_NEAR(rot.determinant(), 1, 1e-6);
  EXPECT_PRED_FORMAT3(AssertMatrixEQ, rot, sol1, 1e-5);

  VectorXd pitch(3);
  pitch << 0, kPi / 4, 0;
  MatrixXd sol2(3, 3);
  EulerToRotation(0, pitch(1), yaw(2), sol2);
  IntegrateSo3(rot, pitch, rot);
  EXPECT_TRUE(rot.isUnitary());
  EXPECT_NEAR(rot.determinant(), 1, 1e-6);
  EXPECT_PRED_FORMAT3(AssertMatrixEQ, rot, sol2, 1e-5);

  VectorXd roll(3);
  roll << kPi / 6, 0, 0;
  MatrixXd sol3(3, 3);
  EulerToRotation(roll(0), roll(1), roll(2), sol3);
  sol3 = sol2 * sol3;
  IntegrateSo3(rot, roll, rot);
  EXPECT_TRUE(rot.isUnitary());
  EXPECT_NEAR(rot.determinant(), 1, 1e-6);
  EXPECT_PRED_FORMAT3(AssertMatrixEQ, rot, sol3, 1e-5);
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

TEST(MatrixFunctionsTest, RandomSpdMatrix) {
  const int n = 5;
  MatrixXd mat(n, n);
  const double eps = 1e-2;
  RandomSpdMatrix(mat, eps);

  EXPECT_TRUE(IsPositiveDefinite(mat));
}

}  // namespace
