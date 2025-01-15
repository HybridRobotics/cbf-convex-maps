#ifndef SCCBF_UTILS_MATRIX_UTILS_H_
#define SCCBF_UTILS_MATRIX_UTILS_H_

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

inline void HatMap(const VectorXd& vec, MatrixXd& vec_hat) {
  const int dim = static_cast<int>(vec_hat.rows());
  assert((dim == 2) || (dim == 3));

  if (dim == 2)
    HatMap<2>(vec, vec_hat);
  else
    HatMap<3>(vec, vec_hat);
}

inline void EulerToRotation(double angle_x, double angle_y, double angle_z,
                            MatrixXd& rotation) {
  assert((rotation.rows() == 3) && (rotation.cols() == 3));
  const Eigen::AngleAxisd roll(angle_x, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch(angle_y, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw(angle_z, Eigen::Vector3d::UnitZ());
  rotation = (yaw * pitch * roll).matrix();
}

inline void RotationFromZVector(const VectorXd& z, MatrixXd& rot) {
  assert(z.rows() == 3);
  assert((rot.rows() == 3) && (rot.cols() == 3));
  assert(z(2) > 0);

  const double norm = z.norm();
  assert(norm > 1e-3);
  const auto zn = z / norm;
  const double sin_x = -zn(1);
  const double cos_x = std::sqrt(zn(0) * zn(0) + zn(2) * zn(2));
  const double sin_y = zn(0) / cos_x;
  const double cos_y = zn(2) / cos_x;

  rot(0, 0) = cos_y;
  rot(1, 0) = 0.0;
  rot(2, 0) = -sin_y;
  rot(0, 1) = sin_x * sin_y;
  rot(1, 1) = cos_x;
  rot(2, 1) = cos_y * sin_x;
  rot.col(2) = zn;
}

inline void AngVelFromZdot(const VectorXd& z, const VectorXd& dz, MatrixXd& rot,
                           VectorXd& wg) {
  assert(dz.rows() == 3);
  assert(wg.rows() == 3);

  RotationFromZVector(z, rot);
  const auto dzn = (dz - z * z.dot(dz) / z.dot(z)) / z.norm();
  const double dphi = -dzn.dot(rot.col(1));
  const double dtheta = dzn.dot(rot.col(0)) / rot(1, 1);
  wg = rot.col(0) * dphi;
  wg(1) = wg(1) + dtheta;
}

template <int dim>
inline void RandomRotation(MatrixXd& rotation) {
  static_assert((dim == 2) || (dim == 3));
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

inline void RandomRotation(MatrixXd& rotation) {
  const int dim = static_cast<int>(rotation.rows());
  assert((dim == 2) || (dim == 3));

  if (dim == 2)
    RandomRotation<2>(rotation);
  else
    RandomRotation<3>(rotation);
}

inline void IntegrateSo2(const MatrixXd& rot, double ang_vel,
                         MatrixXd& rot_new) {
  assert((rot.rows() == 2) && (rot.cols() == 2));
  assert((rot_new.rows() == 2) && (rot_new.cols() == 2));
  const double ang = std::atan2(rot(1, 0), rot(0, 0)) + ang_vel;
  rot_new << std::cos(ang), -std::sin(ang), std::sin(ang), std::cos(ang);
}

inline void ProjectSo3(MatrixXd& rot) {
  rot.col(0).normalize();
  rot.col(1) = rot.col(1) - rot.col(0) * rot.col(0).dot(rot.col(1));
  rot.col(1).normalize();
  rot.col(2) = rot.col(2) - rot.col(0) * rot.col(0).dot(rot.col(2)) -
               rot.col(1) * rot.col(1).dot(rot.col(2));
  rot.col(2).normalize();
}

inline void IntegrateSo3(const MatrixXd& rot, const VectorXd& ang_vel,
                         MatrixXd& rot_new) {
  assert(ang_vel.rows() == 3);
  assert((rot.rows() == 3) && (rot.cols() == 3));
  assert((rot_new.rows() == 3) && (rot_new.cols() == 3));

  const double theta = ang_vel.norm();
  if (theta <= 1e-9) {
    rot_new = rot;
    return;
  }
  const auto dir = ang_vel.normalized();
  MatrixXd hat = MatrixXd::Zero(3, 3);
  HatMap(dir, hat);
  MatrixXd rot_delta = MatrixXd::Identity(3, 3) + std::sin(theta) * hat +
                       (1 - std::cos(theta)) * hat * hat;
  // ProjectSo3(rot_delta);
  rot_new = rot * rot_delta;
  ProjectSo3(rot_new);
}

inline void IntegrateSo(const MatrixXd& rot, const VectorXd& ang_vel,
                        MatrixXd& rot_new) {
  const int n = static_cast<int>(ang_vel.rows());
  assert((n == 1) || (n == 3));
  if (n == 1)
    return IntegrateSo2(rot, ang_vel(0), rot_new);
  else
    return IntegrateSo3(rot, ang_vel, rot_new);
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

template <typename Derived>
inline bool IsPositiveDefinite(const Eigen::MatrixBase<Derived>& mat) {
  assert(mat.rows() == mat.cols());
  const Eigen::LDLT<MatrixXd> ldlt(mat);
  return (ldlt.info() != Eigen::NumericalIssue) && ldlt.isPositive();
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

}  // namespace sccbf

#endif  // SCCBF_UTILS_MATRIX_UTILS_H_
