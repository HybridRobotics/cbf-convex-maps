#ifndef SCCBF_UTILS_H_
#define SCCBF_UTILS_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>
#include <cmath>
#include <random>
#include <string>

#include "sccbf/data_types.h"

namespace sccbf {

// res should be set to zero along the diagonal.
template <int dim, typename Derived>
inline void hat_map(const Eigen::MatrixBase<Derived>& vec, MatrixXd& res) {
  static_assert((dim == 2) || (dim == 3));
  assert(vec.rows() == ((dim == 2) ? 1 : 3));
  assert(vec.cols() == 1);
  assert((res.rows() == dim) && (res.cols() == dim));

  if constexpr (dim == 2) {
    res(1, 0) = vec(0);
    res(0, 1) = -vec(0);
  }
  if constexpr (dim == 3) {
    res(1, 0) = vec(2);
    res(0, 1) = -vec(2);
    res(2, 0) = -vec(1);
    res(0, 2) = vec(1);
    res(2, 1) = vec(0);
    res(1, 2) = -vec(0);
  }
}

// Implementation from "P. Blanchard, D. J. Higham, and N. J. Higham.
// [[https://doi.org/10.1093/imanum/draa038][Computing the Log-Sum-Exp and
// Softmax Functions]]. IMA J. Numer. Anal., Advance access, 2020."
template <typename Derived>
inline double log_sum_exp(const Eigen::MatrixBase<Derived>& vec,
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

inline void euler_to_rot(double ang_x, double ang_y, double ang_z,
                         MatrixXd& rot) {
  assert((rot.rows() == 3) && (rot.cols() == 3));
  const Eigen::AngleAxisd roll(ang_x, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch(ang_y, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw(ang_z, Eigen::Vector3d::UnitZ());
  rot = (yaw * pitch * roll).matrix();
}

template <int dim>
inline MatrixXd random_rotation() {
  std::random_device rd;
  std::mt19937 gen(rd());
  const double pi = static_cast<double>(EIGEN_PI);
  std::uniform_real_distribution<double> dis(-pi, pi);
  MatrixXd rot(dim, dim);
  if constexpr (dim == 2) {
    const double ang = dis(gen);
    rot << std::cos(ang), -std::sin(ang), std::sin(ang), std::cos(ang);
  }
  if constexpr (dim == 3) {
    const double ang_x = dis(gen);
    const double ang_y = dis(gen) / 2.0;
    const double ang_z = dis(gen);
    euler_to_rot(ang_x, ang_y, ang_z, rot);
  }
  return rot;
}

template <typename Derived>
inline bool is_positive_definite(const Eigen::MatrixBase<Derived>& mat) {
  assert(mat.rows() == mat.cols());
  const Eigen::LDLT<MatrixXd> ldlt(mat);
  return (ldlt.info() != Eigen::NumericalIssue) && ldlt.isPositive();
}

}  // namespace sccbf

#endif  // SCCBF_UTILS_H_
