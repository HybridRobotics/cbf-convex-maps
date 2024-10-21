#ifndef SCCBF_DERIVATIVES_H_
#define SCCBF_DERIVATIVES_H_

#include <Eigen/Core>
#include <cassert>

#include "sccbf/data_types.h"

namespace sccbf {

template <int nz>
struct Derivatives {
  VectorXd A;
  VectorXd A_x;
  MatrixXd A_z;
  Vectord<nz> A_xz_y;
  Matrixd<nz, nz> A_zz_y;

  Derivatives(int nr);

  Derivatives(const VectorXd& A_, const VectorXd& A_x_, const MatrixXd& A_z_,
              const Vectord<nz>& A_xz_y_, const Matrixd<nz, nz> A_zz_y);
};

template <int nz>
Derivatives<nz>::Derivatives(int nr) : A(nr), A_x(nr), A_z(nz, nr) {
  A = VectorXd::Zero(nr);
  A_x = VectorXd::Zero(nr);
  A_z = MatrixXd::Zero(nr, nz);
  A_xz_y = VectorXd::Zero(nz);
  A_zz_y = MatrixXd::Zero(nz, nz);
}

template <int nz>
Derivatives<nz>::Derivatives(const VectorXd& A_, const VectorXd& A_x_,
                             const MatrixXd& A_z_, const Vectord<nz>& A_xz_y_,
                             const Matrixd<nz, nz> A_zz_y_)
    : A(A_), A_x(A_x_), A_z(A_z_), A_xz_y(A_xz_y_), A_zz_y(A_zz_y_) {
  const int nr = static_cast<int>(A_.rows());
  assert(A_x_.rows() == nr);
  assert(A_z_.rows() == nr);
  assert(A_z_.cols() == nz);
}

enum class DFlags : uint8_t {
  A = 1 << 0,
  A_x = 1 << 1,
  A_z = 1 << 2,
  A_xz_y = 1 << 3,
  A_zz_y = 1 << 4,
};

inline DFlags operator|(DFlags a, DFlags b) {
  return static_cast<DFlags>(
      static_cast<std::underlying_type<DFlags>::type>(a) |
      static_cast<std::underlying_type<DFlags>::type>(b));
}

inline DFlags operator&(DFlags a, DFlags b) {
  return static_cast<DFlags>(
      static_cast<std::underlying_type<DFlags>::type>(a) &
      static_cast<std::underlying_type<DFlags>::type>(b));
}

inline DFlags operator^(DFlags a, DFlags b) {
  return static_cast<DFlags>(
      static_cast<std::underlying_type<DFlags>::type>(a) ^
      static_cast<std::underlying_type<DFlags>::type>(b));
}

inline DFlags& operator|=(DFlags& a, DFlags b) {
  a = a | b;
  return a;
}

inline DFlags& operator&=(DFlags& a, DFlags b) {
  a = a & b;
  return a;
}

inline DFlags& operator^=(DFlags& a, DFlags b) {
  a = a ^ b;
  return a;
}

inline bool has_dflag(DFlags f1, DFlags f2) {
  return static_cast<std::underlying_type<DFlags>::type>(f1 & f2);
}

}  // namespace sccbf

#endif  // SCCBF_DERIVATIVES_H_
