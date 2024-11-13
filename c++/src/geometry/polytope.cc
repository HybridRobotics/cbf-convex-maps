#include "sccbf/geometry/polytope.h"

#include <Eigen/Core>
#include <cassert>
#include <stdexcept>
#include <string>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

namespace {

constexpr double kPolytopeScThreshold = 1e-2;

}  // namespace

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
  if ((b.array() >= 0).any()) {
    std::runtime_error("Origin is not in the interior of the poytope!");
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
  if (has_flag(flag, DerivativeFlags::f_zz_y) ||
      has_flag(flag, DerivativeFlags::f_zz_y_lb)) {
    derivatives_.f_zz_y =
        2 * sc_modulus_ * y.sum() * MatrixXd::Identity(kNz, kNz);
    derivatives_.f_zz_y_lb = derivatives_.f_zz_y;
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

template class Polytope<2>;
template class Polytope<3>;

}  // namespace sccbf
