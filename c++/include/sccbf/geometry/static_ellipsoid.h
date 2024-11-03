#ifndef SCCBF_GEOMETRY_STATIC_ELLIPSOID_H_
#define SCCBF_GEOMETRY_STATIC_ELLIPSOID_H_

#include <Eigen/Core>
#include <cassert>
#include <stdexcept>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/math_utils/utils.h"

namespace sccbf {

template <int nz_>
class StaticEllipsoid : public ConvexSet {
 public:
  StaticEllipsoid(const MatrixXd& Q, const VectorXd& p, double margin);

  ~StaticEllipsoid();

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
  static constexpr int kNx = 0;
  static constexpr int kNdx = 0;
  static constexpr int kNr = 1;

  const MatrixXd Q_;
  const VectorXd p_;
};

template <int nz_>
StaticEllipsoid<nz_>::StaticEllipsoid(const MatrixXd& Q, const VectorXd& p,
                                      double margin)
    : ConvexSet(kNz, kNr, kNx, kNdx, margin), Q_(Q), p_(p) {
  static_assert(kNz >= 2);
  assert((Q.rows() == kNz) && (Q.cols() == kNz));
  assert(p.rows() == kNz);
  if (!Q.isApprox(Q.transpose())) {
    throw std::runtime_error("Q is not symmetric!");
  }
  if (!IsPositiveDefinite(Q)) {
    throw std::runtime_error("Q is not positive definite!");
  }

  CheckDimensions();
}

template <int nz_>
StaticEllipsoid<nz_>::~StaticEllipsoid() {}

template <int nz_>
inline const Derivatives& StaticEllipsoid<nz_>::UpdateDerivatives(
    const VectorXd& x, const VectorXd& dx, const VectorXd& z, const VectorXd& y,
    DerivativeFlags flag) {
  assert(x.rows() == kNx);
  assert(dx.rows() == kNdx);
  assert(z.rows() == kNz);
  assert(y.rows() == kNr);

  if (has_flag(flag, DerivativeFlags::f)) {
    derivatives_.f(0) = (z - p_).transpose() * Q_ * (z - p_) - 1;
  }
  if (has_flag(flag, DerivativeFlags::f_z)) {
    derivatives_.f_z = 2 * (z - p_).transpose() * Q_;
  }
  if (has_flag(flag, DerivativeFlags::f_zz_y)) {
    derivatives_.f_zz_y = 2 * y(0) * Q_;
  }
  return derivatives_;
}

template <int nz_>
inline void StaticEllipsoid<nz_>::LieDerivatives(const VectorXd& x,
                                                 const VectorXd& z,
                                                 const VectorXd& y,
                                                 const MatrixXd& fg,
                                                 MatrixXd& L_fg_y) const {
  assert(x.rows() == kNx);
  assert(z.rows() == kNz);
  assert(y.rows() == kNr);
  assert(fg.rows() == kNdx);
  assert(L_fg_y.rows() == 1);
  assert(L_fg_y.cols() == fg.cols());

  L_fg_y = MatrixXd::Zero(1, L_fg_y.cols());
}

template <int nz_>
inline int StaticEllipsoid<nz_>::dim() const {
  return kNDim;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::nz() const {
  return kNz;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::nr() const {
  return kNr;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::nx() const {
  return kNx;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::ndx() const {
  return kNdx;
}

template <int nz_>
inline MatrixXd StaticEllipsoid<nz_>::get_projection_matrix() const {
  return MatrixXd::Identity(kNz, kNz);
}

template <int nz_>
inline MatrixXd StaticEllipsoid<nz_>::get_hessian_sparsity_matrix() const {
  return Q_;
}

template <int nz_>
inline bool StaticEllipsoid<nz_>::is_strongly_convex() const {
  return true;
}

typedef StaticEllipsoid<2> StaticEllipsoid2d;
typedef StaticEllipsoid<3> StaticEllipsoid3d;

}  // namespace sccbf

#endif  // SCCBF_GEOMETRY_STATIC_ELLIPSOID_H_
