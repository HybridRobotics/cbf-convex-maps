#ifndef SCCBF_UTILS_H_
#define SCCBF_UTILS_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cassert>
#include <cmath>
#include <random>
#include <string>

#include "sccbf/data_types.h"

namespace sccbf {

// res should be set to zero along the diagonal.
template <int dim, typename Derived>
inline void HatMap(const Eigen::MatrixBase<Derived>& vec, MatrixXd& vec_hat) {
  static_assert((dim == 2) || (dim == 3));
  assert(vec.rows() == ((dim == 2) ? 1 : 3));
  assert(vec.cols() == 1);
  assert((vec_hat.rows() == dim) && (vec_hat.cols() == dim));

  if constexpr (dim == 2) {
    vec_hat(1, 0) = vec(0);
    vec_hat(0, 1) = -vec(0);
  }
  if constexpr (dim == 3) {
    vec_hat(1, 0) = vec(2);
    vec_hat(0, 1) = -vec(2);
    vec_hat(2, 0) = -vec(1);
    vec_hat(0, 2) = vec(1);
    vec_hat(2, 1) = vec(0);
    vec_hat(1, 2) = -vec(0);
  }
}

// Implementation from "P. Blanchard, D. J. Higham, and N. J. Higham.
// [[https://doi.org/10.1093/imanum/draa038][Computing the Log-Sum-Exp and
// Softmax Functions]]. IMA J. Numer. Anal., Advance access, 2020."
template <typename Derived>
inline double LogSumExp(const Eigen::MatrixBase<Derived>& vec,
                        VectorXd& softmax) {
  assert(vec.size() == softmax.size());

  Eigen::Index max_idx{};
  const double max_value = vec.maxCoeff(&max_idx);
  softmax.array() = (vec.array() - max_value).exp();
  softmax(max_idx) = 0;
  const double sum = softmax.sum();
  const double lse = max_value + std::log1p(sum);
  softmax(max_idx) = 1;
  softmax = softmax / (1 + sum);
  return lse;
}

inline void EulerToRotation(double angle_x, double angle_y, double angle_z,
                            MatrixXd& rotation) {
  assert((rotation.rows() == 3) && (rotation.cols() == 3));
  const Eigen::AngleAxisd roll(angle_x, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch(angle_y, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw(angle_z, Eigen::Vector3d::UnitZ());
  rotation = (yaw * pitch * roll).matrix();
}

template <int dim>
inline void RandomRotation(MatrixXd& rotation) {
  assert((rotation.rows() == dim) && (rotation.cols() == dim));

  std::random_device rd;
  std::mt19937 gen(rd());
  const double kPi = static_cast<double>(EIGEN_PI);
  std::uniform_real_distribution<double> dis(-kPi, kPi);

  if constexpr (dim == 2) {
    const double angle = dis(gen);
    rotation(0, 0) = std::cos(angle);
    rotation(0, 1) = -std::sin(angle);
    rotation(1, 0) = std::sin(angle);
    rotation(1, 1) = std::cos(angle);
  }
  if constexpr (dim == 3) {
    const double angle_x = dis(gen);
    const double angle_y = dis(gen) / 2.0;
    const double angle_z = dis(gen);
    EulerToRotation(angle_x, angle_y, angle_z, rotation);
  }
}

inline void RandomSpdMatrix(MatrixXd& mat, const double eps) {
  const int n = static_cast<int>(mat.rows());
  assert(mat.cols() == n);

  const MatrixXd sqrt_mat = MatrixXd::Random(n, n);
  mat = sqrt_mat.transpose() * sqrt_mat + eps * MatrixXd::Identity(n, n);
}

inline void RandomPolytope(const VectorXd& c, double in_radius, MatrixXd& A,
                           VectorXd& b) {
  const int nz = static_cast<int>(c.rows());
  const int nr = static_cast<int>(A.rows());
  assert(b.rows() == nr);
  assert(A.cols() == nz);
  assert(in_radius >= 0);

  for (int i = 0; i < nr;) {
    VectorXd normal = VectorXd::Random(nz);
    if (normal.norm() < 1e-4) continue;
    normal.normalize();
    A.row(i) = normal.transpose();
    b(i) = normal.transpose() * c + in_radius;
    ++i;
  }
}

template <typename Derived>
inline bool IsPositiveDefinite(const Eigen::MatrixBase<Derived>& mat) {
  assert(mat.rows() == mat.cols());
  const Eigen::LDLT<MatrixXd> ldlt(mat);
  return (ldlt.info() != Eigen::NumericalIssue) && ldlt.isPositive();
}

}  // namespace sccbf

#endif  // SCCBF_UTILS_H_
