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
  StaticEllipsoid(const MatrixXd& Q_, const VectorXd& p_, double margin_);

  ~StaticEllipsoid();

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
  static constexpr int nx_ = 0;
  static constexpr int ndx_ = 0;
  static constexpr int nr_ = 1;

  static const MatrixXd proj_mat;
  static constexpr bool strongly_convex = true;

  const MatrixXd Q;
  const VectorXd p;
};

template <int nz_>
const MatrixXd StaticEllipsoid<nz_>::proj_mat = MatrixXd::Identity(nz_, nz_);

template <int nz_>
StaticEllipsoid<nz_>::StaticEllipsoid(const MatrixXd& Q_, const VectorXd& p_,
                                      double margin_)
    : ConvexSet(nz_, nr_, margin_), Q(Q_), p(p_) {
  static_assert(nz_ >= 2);
  assert((Q_.rows() == nz_) && (Q_.cols() == nz_));
  assert(p_.rows() == nz_);
  if (!Q_.isApprox(Q_.transpose())) {
    throw std::runtime_error("Q is not symmetric!");
  }
  if (!is_positive_definite(Q_)) {
    throw std::runtime_error("Q is not positive definite!");
  }
  assert(margin_ >= 0);
}

template <int nz_>
StaticEllipsoid<nz_>::~StaticEllipsoid() {}

template <int nz_>
inline const Derivatives& StaticEllipsoid<nz_>::update_derivatives(
    const VectorXd& x, const VectorXd& dx, const VectorXd& z, const VectorXd& y,
    DFlags f) {
  assert(x.rows() == nx_);
  assert(dx.rows() == ndx_);
  assert(z.rows() == nz_);
  assert(y.rows() == nr_);

  if (has_dflag(f, DFlags::A)) {
    derivatives.A(0) = (z - p).transpose() * Q * (z - p) - 1;
  }
  if (has_dflag(f, DFlags::A_z)) {
    derivatives.A_z = 2 * (z - p).transpose() * Q;
  }
  if (has_dflag(f, DFlags::A_zz_y)) {
    derivatives.A_zz_y = 2 * y(0) * Q;
  }
  return derivatives;
}

template <int nz_>
inline void StaticEllipsoid<nz_>::lie_derivatives_x(const VectorXd& x,
                                                    const VectorXd& z,
                                                    const VectorXd& y,
                                                    const MatrixXd& fg,
                                                    MatrixXd& L_fgA_y) const {
  assert(x.rows() == nx_);
  assert(z.rows() == nz_);
  assert(y.rows() == nr_);
  assert(fg.rows() == ndx_);
  assert(L_fgA_y.rows() == 1);
  assert(L_fgA_y.cols() == fg.cols());

  L_fgA_y = MatrixXd::Zero(1, L_fgA_y.cols());
}

template <int nz_>
inline int StaticEllipsoid<nz_>::dim() const {
  return nz_;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::nz() const {
  return nz_;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::nr() const {
  return nr_;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::nx() const {
  return nx_;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::ndx() const {
  return ndx_;
}

template <int nz_>
inline const MatrixXd& StaticEllipsoid<nz_>::projection_matrix() const {
  return proj_mat;
}

template <int nz_>
inline const MatrixXd& StaticEllipsoid<nz_>::hessian_sparsity_pattern() const {
  return Q;
}

template <int nz_>
inline bool StaticEllipsoid<nz_>::is_strongly_convex() const {
  return strongly_convex;
}

typedef StaticEllipsoid<2> StaticEllipsoid2d;
typedef StaticEllipsoid<3> StaticEllipsoid3d;

}  // namespace sccbf

#endif  // SCCBF_GEOMETRY_STATIC_ELLIPSOID_H_
