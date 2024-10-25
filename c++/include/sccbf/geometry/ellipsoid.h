#ifndef SCCBF_GEOMETRY_ELLIPSOID_H_
#define SCCBF_GEOMETRY_ELLIPSOID_H_

#include <Eigen/Core>
#include <cassert>
#include <stdexcept>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/math_utils/utils.h"

namespace sccbf {

template <int nz_>
class Ellipsoid : public ConvexSet {
 public:
  Ellipsoid(const MatrixXd& Q_, double margin_);

  ~Ellipsoid();

  const Derivatives& update_derivatives(const VectorXd& x, const VectorXd& dx,
                                        const VectorXd& z, const VectorXd& y,
                                        DFlags f) override;

  void lie_derivatives_x(const VectorXd& x, const VectorXd& z,
                         const VectorXd& y, const MatrixXd& fg,
                         MatrixXd& L_fgA_y) const override;

  int dim() const override;

  int nz() const override;

  int nr() const override;

  int nx() const override;

  int ndx() const override;

  const MatrixXd& projection_matrix() const override;

  const MatrixXd& hessian_sparsity_pattern() const override;

  bool is_strongly_convex() const override;

 private:
  static constexpr int nx_ = nz_ + nz_ * nz_;
  static constexpr int ndx_ = (nz_ == 2) ? 3 : 2 * nz_;
  static constexpr int nr_ = 1;

  static const MatrixXd proj_mat;
  static const MatrixXd hess_sparsity;
  static constexpr bool strongly_convex = true;

  const MatrixXd Q;
};

template <int nz_>
const MatrixXd Ellipsoid<nz_>::proj_mat = MatrixXd::Identity(nz_, nz_);

template <int nz_>
const MatrixXd Ellipsoid<nz_>::hess_sparsity = MatrixXd::Ones(nz_, nz_);

template <int nz_>
Ellipsoid<nz_>::Ellipsoid(const MatrixXd& Q_, double margin_)
    : ConvexSet(nz_, nr_, margin_), Q(Q_) {
  static_assert((nz_ == 2) || (nz_ == 3));
  assert((Q_.rows() == nz_) && (Q_.cols() == nz_));
  if (!Q_.isApprox(Q_.transpose())) {
    throw std::runtime_error("Q is not symmetric!");
  }
  if (!is_positive_definite(Q_)) {
    throw std::runtime_error("Q is not positive definite!");
  }

  check_dimensions();
}

template <int nz_>
Ellipsoid<nz_>::~Ellipsoid() {}

template <int nz_>
const Derivatives& Ellipsoid<nz_>::update_derivatives(const VectorXd& x,
                                                      const VectorXd& dx,
                                                      const VectorXd& z,
                                                      const VectorXd& y,
                                                      DFlags f) {
  assert(x.rows() == nx_);
  assert(dx.rows() == ndx_);
  assert(z.rows() == nz_);
  assert(y.rows() == nr_);

  const auto p = x.head<nz_>();
  const auto R = x.tail<nz_ * nz_>().reshaped(nz_, nz_);
  const auto v = dx.head<nz_>();
  MatrixXd wg_hat = MatrixXd::Zero(nz_, nz_);
  if constexpr (nz_ == 2) {
    hat_map<2>(dx.tail<1>(), wg_hat);
  }
  if constexpr (nz_ == 3) {
    hat_map<3>(R * dx.tail<3>(), wg_hat);
  }
  const auto RQRt = R * Q * R.transpose();

  if (has_dflag(f, DFlags::A)) {
    derivatives.A(0) = (z - p).transpose() * RQRt * (z - p) - 1;
  }
  if (has_dflag(f, DFlags::A_z)) {
    derivatives.A_z = 2 * (z - p).transpose() * RQRt;
  }
  if (has_dflag(f, DFlags::A_x)) {
    derivatives.A_x(0) =
        -2 * (z - p).transpose() * RQRt * (v + wg_hat * (z - p));
  }
  if (has_dflag(f, DFlags::A_zz_y)) {
    derivatives.A_zz_y = 2 * y(0) * RQRt;
  }
  if (has_dflag(f, DFlags::A_xz_y)) {
    derivatives.A_xz_y =
        -2 * y(0) * (RQRt * (v + wg_hat * (z - p)) - wg_hat * RQRt * (z - p));
  }
  return derivatives;
}

template <int nz_>
void Ellipsoid<nz_>::lie_derivatives_x(const VectorXd& x, const VectorXd& z,
                                       const VectorXd& y, const MatrixXd& fg,
                                       MatrixXd& L_fgA_y) const {
  assert(x.rows() == nx_);
  assert(z.rows() == nz_);
  assert(y.rows() == nr_);
  assert(fg.rows() == ndx_);
  assert(L_fgA_y.rows() == 1);
  assert(L_fgA_y.cols() == fg.cols());

  const auto p = x.head<nz_>();
  const auto R = x.tail<nz_ * nz_>().reshaped(nz_, nz_);
  const auto RQRt = R * Q * R.transpose();

  if constexpr (nz_ == 2) {
    VectorXd pz_perp(2);
    pz_perp << -(z - p)(1), (z - p)(0);
    L_fgA_y = -2 * y(0) * (z - p).transpose() * RQRt *
              (fg.topRows<2>() + pz_perp * fg.bottomRows<1>());
  }
  if constexpr (nz_ == 3) {
    MatrixXd pz_hat = MatrixXd::Zero(3, 3);
    hat_map<3>(z - p, pz_hat);
    L_fgA_y = -2 * y(0) * (z - p).tranpose() * RQRt *
              (fg.topRows<3>() - pz_hat * R * fg.bottomRows<3>());
  }
}

template <int nz_>
inline int Ellipsoid<nz_>::dim() const {
  return nz_;
}

template <int nz_>
inline int Ellipsoid<nz_>::nz() const {
  return nz_;
}

template <int nz_>
inline int Ellipsoid<nz_>::nr() const {
  return nr_;
}

template <int nz_>
inline int Ellipsoid<nz_>::nx() const {
  return nx_;
}

template <int nz_>
inline int Ellipsoid<nz_>::ndx() const {
  return ndx_;
}

template <int nz_>
inline const MatrixXd& Ellipsoid<nz_>::projection_matrix() const {
  return proj_mat;
}

template <int nz_>
inline const MatrixXd& Ellipsoid<nz_>::hessian_sparsity_pattern() const {
  return hess_sparsity;
}

template <int nz_>
inline bool Ellipsoid<nz_>::is_strongly_convex() const {
  return strongly_convex;
}

typedef Ellipsoid<2> Ellipsoid2d;
typedef Ellipsoid<3> Ellipsoid3d;

}  // namespace sccbf

#endif  // SCCBF_GEOMETRY_ELLIPSOID_H_
