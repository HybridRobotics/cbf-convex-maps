#include "sccbf/math_utils/numerical_derivatives.h"

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

namespace {

// Redefine h so that x + h is representable. Not using this trick leads to
// large error. The compiler flag -ffast-math evaporates these operations.
// From https://github.com/xiaohongchen1991/numcalc/.
inline double adjust_step_size(double x, double h) {
  const double x_plus = x + h;
  h = x_plus - x;
  // Handle the case x + h == x:
  if (h == 0) {
    h = std::nextafter(x, (std::numeric_limits<double>::max)()) - x;
  }
  return h;
}

}  // namespace

void numerical_gradient(const std::function<double(const VectorXd&)>& func,
                        const VectorXd& x, VectorXd& grad) {
  assert(x.size() == grad.size());

  const int dim = static_cast<int>(x.rows());
  const double eps = (std::numeric_limits<double>::epsilon)();

  VectorXd x_p(dim), x_m(dim);

  for (int i = 0; i < dim; ++i) {
    double h = std::pow(3 * eps, 1.0 / 3.0);
    h = adjust_step_size(x(i), h);

    x_p = x;
    x_p(i) = x(i) + h;
    x_m = x;
    x_m(i) = x(i) - h;
    grad(i) = (func(x_p) - func(x_m)) / (2 * h);
  }
}

Derivatives numerical_derivatives(ConvexSet& set, const VectorXd& x,
                                  const VectorXd& x_dot, const VectorXd& dx,
                                  const VectorXd& z, const VectorXd& y) {
  assert(x.rows() == set.nx());
  assert(x_dot.rows() == set.nx());
  assert(dx.rows() == set.ndx());
  assert(z.rows() == set.nz());
  assert(y.rows() == set.nr());

  VectorXd A(set.nr());
  DFlags f = DFlags::A;
  {
    const Derivatives& d = set.update_derivatives(x, dx, z, y, f);
    A = d.A;
  }

  VectorXd A_x(set.nr());
  for (int i = 0; i < set.nr(); ++i) {
    auto func = [&set, &x, &x_dot, &dx, &z, &y, f,
                 i](const VectorXd& h_) -> double {
      const Derivatives& d =
          set.update_derivatives(x + x_dot * h_(0), dx, z, y, f);
      return d.A(i);
    };
    VectorXd grad_(1);
    VectorXd h_(1);
    h_ << 0;
    numerical_gradient(func, h_, grad_);
    A_x(i) = grad_(0);
  }

  MatrixXd A_z(set.nr(), set.nz());
  for (int i = 0; i < set.nr(); ++i) {
    auto func = [&set, &x, &dx, &y, f, i](const VectorXd& z_) -> double {
      const Derivatives& d = set.update_derivatives(x, dx, z_, y, f);
      return d.A(i);
    };
    VectorXd grad_(set.nz());
    numerical_gradient(func, z, grad_);
    A_z.block(i, 0, 1, set.nz()) = grad_.transpose();
  }

  VectorXd A_xz_y(set.nz());
  f = DFlags::A_x;
  {
    auto func = [&set, &x, &dx, &y, f](const VectorXd& z_) -> double {
      const Derivatives& d = set.update_derivatives(x, dx, z_, y, f);
      return y.transpose() * d.A_x;
    };
    numerical_gradient(func, z, A_xz_y);
  }

  MatrixXd A_zz_y(set.nz(), set.nz());
  f = DFlags::A_z;
  for (int i = 0; i < set.nz(); ++i) {
    auto func = [&set, &x, &dx, &y, f, i](const VectorXd& z_) -> double {
      const Derivatives& d = set.update_derivatives(x, dx, z_, y, f);
      return (y.transpose() * d.A_z)(i);
    };
    VectorXd grad_(set.nz());
    numerical_gradient(func, z, grad_);
    A_zz_y.block(i, 0, 1, set.nz()) = grad_.transpose();
  }

  return Derivatives(A, A_x, A_z, A_xz_y, A_zz_y);
}

void numerical_lie_derivatives_x(ConvexSet& set, const VectorXd& x,
                                 const VectorXd& z, const VectorXd& y,
                                 const MatrixXd& fg_dot, const MatrixXd& fg,
                                 MatrixXd& L_fgA_y) {
  assert(x.rows() == set.nx());
  assert(z.rows() == set.nz());
  assert(y.rows() == set.nr());
  assert(fg_dot.rows() == set.nx());
  assert(fg.rows() == set.ndx());
  assert(fg.cols() == fg_dot.cols());
  assert(L_fgA_y.rows() == 1);
  assert(L_fgA_y.cols() == fg.cols());

  DFlags f = DFlags::A;
  for (int i = 0; i < fg.cols(); ++i) {
    const auto dx = fg.col(i);
    const auto x_dot = fg_dot.col(i);
    auto func = [&set, &x, &x_dot, &dx, &z, &y,
                 f](const VectorXd& h_) -> double {
      const Derivatives& d =
          set.update_derivatives(x + x_dot * h_(0), dx, z, y, f);
      return y.transpose() * d.A;
    };
    VectorXd grad_(1);
    VectorXd h_(1);
    h_ << 0;
    numerical_gradient(func, h_, grad_);
    L_fgA_y(i) = grad_(0);
  }
}

}  // namespace sccbf
