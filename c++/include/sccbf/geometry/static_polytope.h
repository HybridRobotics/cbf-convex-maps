#ifndef SCCBF_GEOMETRY_STATIC_POLYTOPE_H_
#define SCCBF_GEOMETRY_STATIC_POLYTOPE_H_

#include <Eigen/Core>
#include <cassert>
#include <stdexcept>
#include <string>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

template <int nz_>
class StaticPolytope : public ConvexSet {
 public:
  StaticPolytope(const MatrixXd& A_, const VectorXd& b_, const VectorXd& p_, double margin_,
           double sc_modulus, bool normalize);

  ~StaticPolytope();

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

  static const MatrixXd proj_mat;

  MatrixXd A;
  VectorXd b;
  const VectorXd& p;
  const double sc_modulus;
  const int nr_;
  MatrixXd hess_sparsity;
  bool strongly_convex;
};

template <int nz_>
const MatrixXd StaticPolytope<nz_>::proj_mat = MatrixXd::Identity(nz_, nz_);

template <int nz_>
StaticPolytope<nz_>::StaticPolytope(const MatrixXd& A_, const VectorXd& b_, const VectorXd& p_, double margin_,
                        double sc_modulus_, bool normalize)
    : ConvexSet(nz_, A_.rows(), margin_),
      A(A_),
      b(b_),
      p(p_),
      sc_modulus(sc_modulus_),
      nr_(A_.rows()) {
  static_assert(nz_ >= 2);
  assert(A_.rows() == b_.rows());
  assert(A_.cols() == nz_);
  assert(A_.rows() >= 1);
  assert(margin_ >= 0);
  assert(sc_modulus_ >= 0);

  for (int i = 0; i < A_.rows(); ++i) {
    const double normi = A.row(i).norm();
    if (normi <= 1e-4) {
      std::runtime_error("Row " + std::to_string(i) +
                         " of A is close to zero!");
    }
    if (normalize) {
      A.row(i).normalize();
      b(i) = b(i) / normi;
    }
  }

  strongly_convex = (sc_modulus >= 1e-3);
  hess_sparsity = strongly_convex? MatrixXd::Identity(nz_, nz_): MatrixXd::Zero(nz_, nz_);
}

template <int nz_>
StaticPolytope<nz_>::~StaticPolytope() {}

template <int nz_>
const Derivatives& StaticPolytope<nz_>::update_derivatives(const VectorXd& x,
                                                     const VectorXd& dx,
                                                     const VectorXd& z,
                                                     const VectorXd& y,
                                                     DFlags f) {
  assert(x.rows() == nx_);
  assert(dx.rows() == ndx_);
  assert(z.rows() == nz_);
  assert(y.rows() == nr_);

  if (has_dflag(f, DFlags::A)) {
    derivatives.A = A * (z - p) - b +
                    sc_modulus * (z - p).squaredNorm() * VectorXd::Ones(nr_);
  }
  if (has_dflag(f, DFlags::A_z)) {
    derivatives.A_z =
        A + 2 * sc_modulus * VectorXd::Ones(nr_) * (z - p).transpose();
  }
  if (has_dflag(f, DFlags::A_zz_y)) {
    derivatives.A_zz_y =
        2 * sc_modulus * y.sum() * MatrixXd::Identity(nz_, nz_);
  }
  return derivatives;
}

template <int nz_>
void StaticPolytope<nz_>::lie_derivatives_x(const VectorXd& x, const VectorXd& z,
                                      const VectorXd& y, const MatrixXd& fg,
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
inline int StaticPolytope<nz_>::dim() const {
  return nz_;
}

template <int nz_>
inline int StaticPolytope<nz_>::nz() const {
  return nz_;
}

template <int nz_>
inline int StaticPolytope<nz_>::nr() const {
  return nr_;
}

template <int nz_>
inline int StaticPolytope<nz_>::nx() const {
  return nx_;
}

template <int nz_>
inline int StaticPolytope<nz_>::ndx() const {
  return ndx_;
}

template <int nz_>
inline const MatrixXd& StaticPolytope<nz_>::projection_matrix() const {
  return proj_mat;
}

template <int nz_>
inline const MatrixXd& StaticPolytope<nz_>::hessian_sparsity_pattern() const {
  return hess_sparsity;
}

template <int nz_>
inline bool StaticPolytope<nz_>::is_strongly_convex() const {
  return strongly_convex;
}

typedef StaticPolytope<2> StaticPolytope2d;
typedef StaticPolytope<3> StaticPolytope3d;

}  // namespace sccbf

#endif  // SCCBF_GEOMETRY_STATIC_POLYTOPE_H_
