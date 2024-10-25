#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <random>

#include "sccbf/data_types.h"
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

TEST(MatrixFunctions, HatMap2D) {
  VectorXd vec(1);
  vec << 1;
  MatrixXd hat(2, 2);
  hat = MatrixXd::Zero(2, 2);
  MatrixXd sol(2, 2);
  sol << 0, -1, 1, 0;

  hat_map<2>(vec, hat);
  MatrixDoubleEQ(hat, sol);
}

TEST(MatrixFunctions, HatMap3D) {
  VectorXd vec(3);
  vec << 1, 2, 3;
  MatrixXd hat(3, 3);
  hat = MatrixXd::Zero(3, 3);
  MatrixXd sol(3, 3);
  sol << 0, -3, 2, 3, 0, -1, -2, 1, 0;

  hat_map<3>(vec, hat);
  MatrixDoubleEQ(hat, sol);
}

TEST(MatrixFunctions, EulerToRot) {
  // Set random generator.
  std::mt19937 gen(1);
  const double pi = static_cast<double>(EIGEN_PI);
  std::uniform_real_distribution<double> dis(-pi, pi);

  const int num_trials = 10;
  for (int i = 0; i < num_trials; ++i) {
    const double ang_x = dis(gen);
    const double ang_y = dis(gen) / 2.0;
    const double ang_z = dis(gen);
    MatrixXd rot(3, 3);

    euler_to_rot(ang_x, ang_y, ang_z, rot);

    EXPECT_TRUE(rot.isUnitary());
    EXPECT_NEAR(rot.determinant(), 1, 1e-6);
  }
}

TEST(MatrixFunctions, RandomRotation2D) {
  const int num_trials = 10;
  for (int i = 0; i < num_trials; ++i) {
    MatrixXd rot = random_rotation<2>();

    EXPECT_TRUE(rot.isUnitary());
    EXPECT_NEAR(rot.determinant(), 1, 1e-6);
  }
}

TEST(MatrixFunctions, RandomRotation3D) {
  const int num_trials = 10;
  for (int i = 0; i < num_trials; ++i) {
    MatrixXd rot = random_rotation<3>();

    EXPECT_TRUE(rot.isUnitary());
    EXPECT_NEAR(rot.determinant(), 1, 1e-6);
  }
}

TEST(MatrixFunctions, IsPositiveDefinite) {
  const int num_trials = 10;
  for (int i = 0; i < num_trials; ++i) {
    MatrixXd R = random_rotation<3>();
    VectorXd eig(3);
    eig << -1, 0, 1;
    MatrixXd mat = R * eig.asDiagonal() * R.transpose();

    EXPECT_FALSE(is_positive_definite(mat));
  }

  for (int i = 0; i < num_trials; ++i) {
    MatrixXd R = random_rotation<3>();
    VectorXd eig(3);
    eig << 0.01, 0.01, 1;
    MatrixXd mat = R * eig.asDiagonal() * R.transpose();

    EXPECT_TRUE(is_positive_definite(mat));
  }
}

}  // namespace
