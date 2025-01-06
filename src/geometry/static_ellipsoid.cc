#include "sccbf/geometry/static_ellipsoid.h"

#include <Eigen/Core>
#include <cassert>
#include <stdexcept>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

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
  if (has_flag(flag, DerivativeFlags::f_zz_y) ||
      has_flag(flag, DerivativeFlags::f_zz_y_lb)) {
    derivatives_.f_zz_y = 2 * y(0) * Q_;
    derivatives_.f_zz_y_lb = derivatives_.f_zz_y;
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

template class StaticEllipsoid<2>;
template class StaticEllipsoid<3>;
template class StaticEllipsoid<4>;

}  // namespace sccbf
