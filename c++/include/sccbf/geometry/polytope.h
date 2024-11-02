#ifndef SCCBF_GEOMETRY_POLYTOPE_H_
#define SCCBF_GEOMETRY_POLYTOPE_H_

#include <Eigen/Core>
#include <cassert>
#include <stdexcept>
#include <string>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/math_utils/utils.h"

namespace sccbf {

namespace {

constexpr double kPolytopeScThreshold = 1e-3;

}  // namespace

template <int nz_>
class Polytope : public ConvexSet {
 public:
  Polytope(const MatrixXd& A, const VectorXd& b, double margin,
           double sc_modulus, bool normalize);

  ~Polytope();

  const Derivatives& UpdateDerivatives(const VectorXd& x, const VectorXd& dx,
                                       const VectorXd& z, const VectorXd& y,
                                       DerivativeFlags flag) override;

  void LieDerivatives(const VectorXd& x, const VectorXd& z, const VectorXd& y,
                      const MatrixXd& fg, MatrixXd& L_fg_y) const override;

  int dim() const override;

  int nz() const override;

  int nr() const override;

  int nx() const override;

  int ndx() const override;

  MatrixXd get_projection_matrix() const override;

  MatrixXd get_hessian_sparsity_matrix() const override;

  bool is_strongly_convex() const override;

 private:
  static constexpr int kNz = nz_;
  static constexpr int kNDim = kNz;
  static constexpr int kNx = kNz + kNz * kNz;
  static constexpr int kNdx = (kNz == 2) ? 3 : 2 * kNz;

  MatrixXd A_;
  VectorXd b_;
  const double sc_modulus_;
  const int nr_;
  bool strongly_convex_;
};

template <int nz_>
Polytope<nz_>::Polytope(const MatrixXd& A, const VectorXd& b, double margin,
                        double sc_modulus, bool normalize)
    : ConvexSet(kNz, static_cast<int>(A.rows()), kNx, kNdx, margin),
      A_(A),
      b_(b),
      sc_modulus_(sc_modulus),
      nr_(static_cast<int>(A.rows())) {
  static_assert((kNz == 2) || (kNz == 3));
  assert(A.rows() == b.rows());
  assert(A.cols() == kNz);
  assert(sc_modulus >= 0);

  if (A.rows() <= kNz) {
    std::runtime_error("Polytope is not compact!");
  }
  for (int i = 0; i < A.rows(); ++i) {
    const double row_norm = A.row(i).norm();
    if (row_norm <= 1e-4) {
      std::runtime_error("Row " + std::to_string(i) +
                         " of A is close to zero!");
    }
    if (normalize) {
      A_.row(i).normalize();
      b_(i) = b_(i) / row_norm;
    }
  }

  strongly_convex_ = (sc_modulus >= kPolytopeScThreshold);

  CheckDimensions();
}

template <int nz_>
Polytope<nz_>::~Polytope() {}

template <int nz_>
const Derivatives& Polytope<nz_>::UpdateDerivatives(const VectorXd& x,
                                                    const VectorXd& dx,
                                                    const VectorXd& z,
                                                    const VectorXd& y,
                                                    DerivativeFlags flag) {
  assert(x.rows() == kNx);
  assert(dx.rows() == kNdx);
  assert(z.rows() == kNz);
  assert(y.rows() == nr_);

  const auto p = x.head<kNz>();
  const auto R = x.tail<kNz * kNz>().reshaped(kNz, kNz);
  const auto v = dx.head<kNz>();
  MatrixXd wg_hat = MatrixXd::Zero(kNz, kNz);
  if constexpr (kNz == 2) {
    HatMap<2>(dx.tail<1>(), wg_hat);
  }
  if constexpr (kNz == 3) {
    HatMap<3>(R * dx.tail<3>(), wg_hat);
  }
  const auto ARt = A_ * R.transpose();

  if (has_flag(flag, DerivativeFlags::f)) {
    derivatives_.f = ARt * (z - p) - b_ +
                     sc_modulus_ * (z - p).squaredNorm() * VectorXd::Ones(nr_);
  }
  if (has_flag(flag, DerivativeFlags::f_z) ||
      has_flag(flag, DerivativeFlags::f_x)) {
    derivatives_.f_z =
        ARt + 2 * sc_modulus_ * VectorXd::Ones(nr_) * (z - p).transpose();
    derivatives_.f_x = -ARt * wg_hat * (z - p) - derivatives_.f_z * v;
  }
  if (has_flag(flag, DerivativeFlags::f_zz_y)) {
    derivatives_.f_zz_y =
        2 * sc_modulus_ * y.sum() * MatrixXd::Identity(kNz, kNz);
  }
  if (has_flag(flag, DerivativeFlags::f_xz_y)) {
    derivatives_.f_xz_y = -(y.transpose() * ARt * wg_hat).transpose() -
                          2 * sc_modulus_ * y.sum() * v;
  }
  return derivatives_;
}

template <int nz_>
void Polytope<nz_>::LieDerivatives(const VectorXd& x, const VectorXd& z,
                                   const VectorXd& y, const MatrixXd& fg,
                                   MatrixXd& L_fg_y) const {
  assert(x.rows() == kNx);
  assert(z.rows() == kNz);
  assert(y.rows() == nr_);
  assert(fg.rows() == kNdx);
  assert(L_fg_y.rows() == 1);
  assert(L_fg_y.cols() == fg.cols());

  const auto p = x.head<kNz>();
  const auto R = x.tail<kNz * kNz>().reshaped(kNz, kNz);
  const auto yARt = y.transpose() * A_ * R.transpose();

  if constexpr (kNz == 2) {
    VectorXd pz_perp(2);
    pz_perp << -(z - p)(1), (z - p)(0);
    L_fg_y = -yARt * (fg.topRows<2>() + pz_perp * fg.bottomRows<1>()) -
             2 * sc_modulus_ * y.sum() * (z - p).transpose() * fg.topRows<2>();
  }
  if constexpr (kNz == 3) {
    MatrixXd pz_hat = Eigen::MatrixXd(3, 3);
    HatMap<3>(z - p, pz_hat);
    L_fg_y = -yARt * (fg.topRows<3>() - pz_hat * R * fg.bottomRows<3>()) -
             2 * sc_modulus_ * y.sum() * (z - p).transpose() * fg.topRows<3>();
  }
}

template <int nz_>
inline int Polytope<nz_>::dim() const {
  return kNDim;
}

template <int nz_>
inline int Polytope<nz_>::nz() const {
  return kNz;
}

template <int nz_>
inline int Polytope<nz_>::nr() const {
  return nr_;
}

template <int nz_>
inline int Polytope<nz_>::nx() const {
  return kNx;
}

template <int nz_>
inline int Polytope<nz_>::ndx() const {
  return kNdx;
}

template <int nz_>
inline MatrixXd Polytope<nz_>::get_projection_matrix() const {
  return MatrixXd::Identity(kNz, kNz);
}

template <int nz_>
inline MatrixXd Polytope<nz_>::get_hessian_sparsity_matrix() const {
  if (strongly_convex_)
    return MatrixXd::Identity(kNz, kNz);
  else
    return MatrixXd::Zero(kNz, kNz);
}

template <int nz_>
inline bool Polytope<nz_>::is_strongly_convex() const {
  return strongly_convex_;
}

typedef Polytope<2> Polytope2d;
typedef Polytope<3> Polytope3d;

}  // namespace sccbf

#endif  // SCCBF_GEOMETRY_POLYTOPE_H_
