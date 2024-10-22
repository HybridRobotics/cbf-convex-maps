#ifndef SCCBF_UTILS_H_
#define SCCBF_UTILS_H_

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <functional>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

// result should be set to zero along the diagonal.
template <int dim, typename Derived>
inline void hat_map(const Eigen::MatrixBase<Derived>& vec,
                    Matrixd<dim, dim>& res) {
  static_assert((dim == 2) || (dim == 3));
  assert(vec.rows() == ((dim == 2) ? 1 : 3));
  assert(vec.cols() == 1);

  if constexpr (dim == 2) {
    res(1, 0) = vec(0);
    res(0, 1) = -vec(0);
  }
  if constexpr (dim == 3) {
    res(1, 0) = vec(2);
    res(0, 1) = -vec(2);
    res(2, 0) = -vec(1);
    res(0, 2) = vec(1);
    res(2, 1) = vec(0);
    res(1, 2) = -vec(0);
  }
}

// Implementation from "P. Blanchard, D. J. Higham, and N. J. Higham.
// [[https://doi.org/10.1093/imanum/draa038][Computing the Log-Sum-Exp and
// Softmax Functions]]. IMA J. Numer. Anal., Advance access, 2020."
template <typename Derived>
inline double log_sum_exp(const Eigen::MatrixBase<Derived>& vec,
                          VectorXd& softmax) {
  assert(vec.size() == softmax.size());

  Eigen::Index max_idx{};
  const double max_value = vec.maxCoeff(&max_idx);
  softmax.array() = (vec.array() - max_value).exp();
  softmax(max_idx) = 0;
  const double sum = softmax.sum();
  const double lse = max_value + std::log1p(sum);
  softmax(max_idx) = 1;
  softmax = softmax / (1 + sum);
  return lse;
}

void numerical_gradient(const std::function<double(const VectorXd&)>& func,
                        const VectorXd& x, VectorXd& grad);

template <int nz>
Derivatives<nz> numerical_derivatives(const ConvexSet<nz>& set,
                                      const VectorXd& x, const VectorXd& dx,
                                      const Vectord<nz>& z, const VectorXd& y) {
  assert(x.rows() == set.nx());
  assert(dx.rows() == set.ndx());
  assert(y.rows() == set.nr());

  VectorXd A(set.nr());
  DFlags f = DFlags::A;
  {
    const Derivatives<nz>& d = set.update_derivatives(x, dx, z, y, f);
    A = d.A;
  }

  VectorXd A_x(set.nr());
  for (int i = 0; i < set.nr(); ++i) {
    auto func = [&set, &x, &dx, &z, &y, f, i](const VectorXd& h_) -> double {
      const Derivatives<nz>& d =
          set.update_derivatives(x + dx * h_, dx, z, y, f);
      return d.A(i);
    };
    VectorXd grad_(1);
    numerical_gradient(func, {0}, grad_);
    A_x(i) = grad_(0);
  }

  MatrixXd A_z(set.nr(), nz);
  for (int i = 0; i < set.nr(); ++i) {
    auto func = [&set, &x, &dx, &y, f, i](const VectorXd& z_) -> double {
      const Derivatives<nz>& d = set.update_derivatives(x, dx, z_, y, f);
      return d.A(i);
    };
    VectorXd grad_(nz);
    numerical_gradient(func, z, grad_);
    A_z.block<1, nz>(i, 0) = grad_.transpose();
  }

  Vectord<nz> A_xz_y;
  f = DFlags::A_x;
  {
    auto func = [&set, &x, &dx, &y, f](const VectorXd& z_) -> double {
      const Derivatives<nz>& d = set.update_derivatives(x, dx, z_, y, f);
      return y.transpose() * d.A_x;
    };
    numerical_gradient(func, z, A_xz_y);
  }

  Matrixd<nz, nz> A_zz_y;
  f = DFlags::A_z;
  for (int i = 0; i < nz; ++i) {
    auto func = [&set, &x, &dx, &y, f, i](const VectorXd& z_) -> double {
      const Derivatives<nz>& d = set.update_derivatives(x, dx, z_, y, f);
      return (y.transpose() * d.A_z)(i);
    };
    VectorXd grad_(nz);
    numerical_gradient(func, z, grad_);
    A_z.block<1, nz>(i, 0) = grad_.transpose();
  }

  return Derivatives<nz>(A, A_x, A_z, A_xz_y, A_zz_y);
}

template <int nz>
Derivatives<nz> numerical_lie_derivatives_x(
    const ConvexSet<nz>& set, const VectorXd& x, const Vectord<nz>& z,
    const VectorXd& f, const MatrixXd& g, VectorXd& L_fA, MatrixXd& L_gA) {
  assert(x.rows() == set.nx());
  assert(f.rows() == set.ndx());
  assert(g.rows() == set.ndx());
  assert(L_fA.rows() == set.nr());
  assert(L_gA.rows() == set.nr());
  assert(L_gA.cols() == g.cols());

  // TODO
}

}  // namespace sccbf
#endif  // SCCBF_UTILS_H_
