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

template <int nz_>
class Polytope : public ConvexSet {
 public:
  Polytope(const MatrixXd& A_, const VectorXd& b_, double margin_,
           double sc_modulus, bool normalize);

  ~Polytope();

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

  static const MatrixXd proj_mat;

  MatrixXd A;
  VectorXd b;
  const double sc_modulus;
  const int nr_;
  MatrixXd hess_sparsity;
  bool strongly_convex;
};

template <int nz_>
const MatrixXd Polytope<nz_>::proj_mat = MatrixXd::Identity(nz_, nz_);

template <int nz_>
Polytope<nz_>::Polytope(const MatrixXd& A_, const VectorXd& b_, double margin_,
                        double sc_modulus_, bool normalize)
    : ConvexSet(nz_, A_.rows(), margin_),
      A(A_),
      b(b_),
      sc_modulus(sc_modulus_),
      nr_(A_.rows()) {
  static_assert((nz_ == 2) || (nz_ == 3));
  assert(A_.rows() == b_.rows());
  assert(A_.cols() == nz_);
  assert(margin_ >= 0);
  assert(sc_modulus_ >= 0);

  if (A_.rows() <= nz_) {
    std::runtime_error("Polytope is not compact!");
  }
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
  hess_sparsity = strongly_convex? MatrixXd::Identity(nz_, nz_): MatrixXd::Zeros(nz_, nz_);
}

template <int nz_>
Polytope<nz_>::~Polytope() {}

template <int nz_>
const Derivatives& Polytope<nz_>::update_derivatives(const VectorXd& x,
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
  const auto ARt = A * R.transpose();

  if (has_dflag(f, DFlags::A)) {
    derivatives.A = ARt * (z - p) - b +
                    sc_modulus * (z - p).squaredNorm() * VectorXd::Ones(nr_);
  }
  if (has_dflag(f, DFlags::A_z) || has_dflag(f, DFlags::A_x)) {
    derivatives.A_z =
        ARt + 2 * sc_modulus * VectorXd::Ones(nr_) * (z - p).transpose();
    derivatives.A_x = -ARt * wg_hat * (z - p) - derivatives.A_z * v;
  }
  if (has_dflag(f, DFlags::A_zz_y)) {
    derivatives.A_zz_y =
        2 * sc_modulus * y.sum() * MatrixXd::Identity(nz_, nz_);
  }
  if (has_dflag(f, DFlags::A_xz_y)) {
    derivatives.A_xz_y =
        -y.transpose() * ARt * wg_hat - 2 * sc_modulus * y.sum() * v;
  }
  return derivatives;
}

template <int nz_>
void Polytope<nz_>::lie_derivatives_x(const VectorXd& x, const VectorXd& z,
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
  const auto yARt = y.transpose() * A * R.transpose();

  if constexpr (nz_ == 2) {
    VectorXd pz_perp(2);
    pz_perp << -(z - p)(1), (z - p)(0);
    L_fgA_y = -yARt * (fg.topRows<2>() + pz_perp * fg.bottomRows<1>()) -
              2 * sc_modulus * y.sum() * (z - p).transpose() * fg.topRows<2>();
  }
  if constexpr (nz_ == 3) {
    MatrixXd pz_hat = Eigen::MatrixXd(3, 3);
    hat_map<3>(z - p, pz_hat);
    L_fgA_y = -yARt * (fg.topRows<3>() - pz_hat * R * fg.bottomRows<3>()) -
              2 * sc_modulus * y.sum() * (z - p).transpose() * fg.topRows<3>();
  }
}

template <int nz_>
inline int Polytope<nz_>::dim() const {
  return nz_;
}

template <int nz_>
inline int Polytope<nz_>::nz() const {
  return nz_;
}

template <int nz_>
inline int Polytope<nz_>::nr() const {
  return nr_;
}

template <int nz_>
inline int Polytope<nz_>::nx() const {
  return nx_;
}

template <int nz_>
inline int Polytope<nz_>::ndx() const {
  return ndx_;
}

template <int nz_>
inline const MatrixXd& Polytope<nz_>::projection_matrix() const {
  return proj_mat;
}

template <int nz_>
inline const MatrixXd& Polytope<nz_>::hessian_sparsity_pattern() const {
  return hess_sparsity;
}

template <int nz_>
inline bool Polytope<nz_>::is_strongly_convex() const {
  return strongly_convex;
}

typedef Polytope<2> Polytope2d;
typedef Polytope<3> Polytope3d;

}  // namespace sccbf

#endif  // SCCBF_GEOMETRY_POLYTOPE_H_
