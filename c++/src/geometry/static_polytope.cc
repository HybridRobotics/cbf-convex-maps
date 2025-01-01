#include "sccbf/geometry/static_polytope.h"

#include <Eigen/Core>
#include <cassert>
#include <stdexcept>
#include <string>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

namespace {

constexpr double kStaticPolytopeScThreshold = 1e-2;

}  // namespace

template <int nz_>
StaticPolytope<nz_>::StaticPolytope(const MatrixXd& A, const VectorXd& b,
                                    const VectorXd& p, double margin,
                                    double sc_modulus, bool normalize)
    : ConvexSet(kNz, static_cast<int>(A.rows()), kNx, kNdx, margin),
      A_(A),
      b_(b),
      p_(p),
      sc_modulus_(sc_modulus),
      nr_(static_cast<int>(A.rows())) {
  static_assert(kNz >= 1);
  assert(A.rows() == b.rows());
  assert(A.cols() == kNz);
  assert(A.rows() >= 1);
  assert(p.rows() == kNz);
  assert(sc_modulus >= 0);

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
  if (((A * p - b).array() >= 0).any()) {
    std::runtime_error("p is not in the interior of the poytope!");
  }

  strongly_convex_ = (sc_modulus >= kStaticPolytopeScThreshold);

  CheckDimensions();
}

template <int nz_>
StaticPolytope<nz_>::~StaticPolytope() {}

template <int nz_>
const Derivatives& StaticPolytope<nz_>::UpdateDerivatives(
    const VectorXd& x, const VectorXd& dx, const VectorXd& z, const VectorXd& y,
    DerivativeFlags flag) {
  assert(x.rows() == kNx);
  assert(dx.rows() == kNdx);
  assert(z.rows() == kNz);
  assert(y.rows() == nr_);

  if (has_flag(flag, DerivativeFlags::f)) {
    derivatives_.f = A_ * (z - p_) - b_ +
                     sc_modulus_ * (z - p_).squaredNorm() * VectorXd::Ones(nr_);
  }
  if (has_flag(flag, DerivativeFlags::f_z)) {
    derivatives_.f_z =
        A_ + 2 * sc_modulus_ * VectorXd::Ones(nr_) * (z - p_).transpose();
  }
  if (has_flag(flag, DerivativeFlags::f_zz_y) ||
      has_flag(flag, DerivativeFlags::f_zz_y_lb)) {
    derivatives_.f_zz_y =
        2 * sc_modulus_ * y.sum() * MatrixXd::Identity(kNz, kNz);
    derivatives_.f_zz_y_lb = derivatives_.f_zz_y;
  }
  return derivatives_;
}

template <int nz_>
void StaticPolytope<nz_>::LieDerivatives(const VectorXd& x, const VectorXd& z,
                                         const VectorXd& y, const MatrixXd& fg,
                                         MatrixXd& L_fg_y) const {
  assert(x.rows() == kNx);
  assert(z.rows() == kNz);
  assert(y.rows() == nr_);
  assert(fg.rows() == kNdx);
  assert(L_fg_y.rows() == 1);
  assert(L_fg_y.cols() == fg.cols());

  L_fg_y = MatrixXd::Zero(1, L_fg_y.cols());
}

template class StaticPolytope<2>;
template class StaticPolytope<3>;
template class StaticPolytope<4>;

}  // namespace sccbf
