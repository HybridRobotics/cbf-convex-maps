#include "sccbf/geometry/ellipsoid.h"

#include <Eigen/Core>
#include <cassert>
#include <stdexcept>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

template <int nz_>
Ellipsoid<nz_>::Ellipsoid(const MatrixXd& Q, double margin)
    : ConvexSet(kNz, kNr, kNx, kNdx, margin), Q_(Q) {
  static_assert((kNz == 2) || (kNz == 3));
  assert((Q.rows() == kNz) && (Q.cols() == kNz));
  if (!Q.isApprox(Q.transpose())) {
    throw std::runtime_error("Q is not symmetric!");
  }
  if (!IsPositiveDefinite(Q)) {
    throw std::runtime_error("Q is not positive definite!");
  }

  CheckDimensions();
}

template <int nz_>
Ellipsoid<nz_>::~Ellipsoid() {}

template <int nz_>
const Derivatives& Ellipsoid<nz_>::UpdateDerivatives(const VectorXd& x,
                                                     const VectorXd& dx,
                                                     const VectorXd& z,
                                                     const VectorXd& y,
                                                     DerivativeFlags flag) {
  assert(x.rows() == kNx);
  assert(dx.rows() == kNdx);
  assert(z.rows() == kNz);
  assert(y.rows() == kNr);

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
  const auto RQRt = R * Q_ * R.transpose();

  if (has_flag(flag, DerivativeFlags::f)) {
    derivatives_.f(0) = (z - p).transpose() * RQRt * (z - p) - 1;
  }
  if (has_flag(flag, DerivativeFlags::f_z)) {
    derivatives_.f_z = 2 * (z - p).transpose() * RQRt;
  }
  if (has_flag(flag, DerivativeFlags::f_x)) {
    derivatives_.f_x(0) =
        -2 * (z - p).transpose() * RQRt * (v + wg_hat * (z - p));
  }
  if (has_flag(flag, DerivativeFlags::f_zz_y) ||
      has_flag(flag, DerivativeFlags::f_zz_y_lb)) {
    derivatives_.f_zz_y = 2 * y(0) * RQRt;
    derivatives_.f_zz_y_lb = derivatives_.f_zz_y;
  }
  if (has_flag(flag, DerivativeFlags::f_xz_y)) {
    derivatives_.f_xz_y =
        -2 * y(0) * (RQRt * (v + wg_hat * (z - p)) - wg_hat * RQRt * (z - p));
  }
  return derivatives_;
}

template <int nz_>
void Ellipsoid<nz_>::LieDerivatives(const VectorXd& x, const VectorXd& z,
                                    const VectorXd& y, const MatrixXd& fg,
                                    MatrixXd& L_fg_y) const {
  assert(x.rows() == kNx);
  assert(z.rows() == kNz);
  assert(y.rows() == kNr);
  assert(fg.rows() == kNdx);
  assert(L_fg_y.rows() == 1);
  assert(L_fg_y.cols() == fg.cols());

  const auto p = x.head<kNz>();
  const auto R = x.tail<kNz * kNz>().reshaped(kNz, kNz);
  const auto RQRt = R * Q_ * R.transpose();

  if constexpr (kNz == 2) {
    VectorXd pz_perp(2);
    pz_perp << -(z - p)(1), (z - p)(0);
    L_fg_y = -2 * y(0) * (z - p).transpose() * RQRt *
             (fg.topRows<2>() + pz_perp * fg.bottomRows<1>());
  }
  if constexpr (kNz == 3) {
    MatrixXd pz_hat = MatrixXd::Zero(3, 3);
    HatMap<3>(z - p, pz_hat);
    L_fg_y = -2 * y(0) * (z - p).transpose() * RQRt *
             (fg.topRows<3>() - pz_hat * R * fg.bottomRows<3>());
  }
}

template class Ellipsoid<2>;
template class Ellipsoid<3>;

}  // namespace sccbf
