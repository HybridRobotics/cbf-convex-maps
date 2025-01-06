#ifndef SCCBF_DERIVATIVES_H_
#define SCCBF_DERIVATIVES_H_

#include <Eigen/Core>
#include <cassert>
#include <type_traits>

#include "sccbf/data_types.h"

namespace sccbf {

struct Derivatives {
  VectorXd f;
  VectorXd f_x;
  MatrixXd f_z;
  VectorXd f_xz_y;
  MatrixXd f_zz_y;
  MatrixXd f_zz_y_lb;

  Derivatives(int nz, int nr);

  Derivatives(const VectorXd& f, const VectorXd& f_x, const MatrixXd& f_z,
              const VectorXd& f_xz_y, const MatrixXd& f_zz_y);
};

inline Derivatives::Derivatives(int nz, int nr)
    : f(nr),
      f_x(nr),
      f_z(nr, nz),
      f_xz_y(nz),
      f_zz_y(nz, nz),
      f_zz_y_lb(nz, nz) {
  f = VectorXd::Zero(nr);
  f_x = VectorXd::Zero(nr);
  f_z = MatrixXd::Zero(nr, nz);
  f_xz_y = VectorXd::Zero(nz);
  f_zz_y = MatrixXd::Zero(nz, nz);
  f_zz_y_lb = MatrixXd::Zero(nz, nz);
}

inline Derivatives::Derivatives(const VectorXd& f, const VectorXd& f_x,
                                const MatrixXd& f_z, const VectorXd& f_xz_y,
                                const MatrixXd& f_zz_y)
    : f(f), f_x(f_x), f_z(f_z), f_xz_y(f_xz_y), f_zz_y(f_zz_y) {
  const int nz = static_cast<int>(f_z.cols());
  const int nr = static_cast<int>(f.rows());
  assert(f_x.rows() == nr);
  assert(f_z.rows() == nr);
  assert(f_xz_y.rows() == nz);
  assert((f_zz_y.rows() == nz) && (f_zz_y.cols() == nz));
}

enum class DerivativeFlags : uint8_t {
  f = 1 << 0,
  f_x = 1 << 1,
  f_z = 1 << 2,
  f_xz_y = 1 << 3,
  f_zz_y = 1 << 4,
  f_zz_y_lb = 1 << 5,
};

inline DerivativeFlags operator|(DerivativeFlags a, DerivativeFlags b) {
  return static_cast<DerivativeFlags>(
      static_cast<std::underlying_type<DerivativeFlags>::type>(a) |
      static_cast<std::underlying_type<DerivativeFlags>::type>(b));
}

inline DerivativeFlags operator&(DerivativeFlags a, DerivativeFlags b) {
  return static_cast<DerivativeFlags>(
      static_cast<std::underlying_type<DerivativeFlags>::type>(a) &
      static_cast<std::underlying_type<DerivativeFlags>::type>(b));
}

inline DerivativeFlags operator^(DerivativeFlags a, DerivativeFlags b) {
  return static_cast<DerivativeFlags>(
      static_cast<std::underlying_type<DerivativeFlags>::type>(a) ^
      static_cast<std::underlying_type<DerivativeFlags>::type>(b));
}

inline DerivativeFlags& operator|=(DerivativeFlags& a, DerivativeFlags b) {
  a = a | b;
  return a;
}

inline DerivativeFlags& operator&=(DerivativeFlags& a, DerivativeFlags b) {
  a = a & b;
  return a;
}

inline DerivativeFlags& operator^=(DerivativeFlags& a, DerivativeFlags b) {
  a = a ^ b;
  return a;
}

inline bool has_flag(DerivativeFlags flag, DerivativeFlags a) {
  return static_cast<std::underlying_type<DerivativeFlags>::type>(flag & a);
}

}  // namespace sccbf

#endif  // SCCBF_DERIVATIVES_H_
