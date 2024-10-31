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
inline double AdjustStepSize(double x, double h) {
  const double x_plus = x + h;
  h = x_plus - x;
  // Handle the case x + h == x:
  if (h == 0) {
    h = std::nextafter(x, (std::numeric_limits<double>::max)()) - x;
  }
  return h;
}

}  // namespace

void NumericalGradient(const std::function<double(const VectorXd&)>& func,
                       const VectorXd& x, VectorXd& grad) {
  assert(x.size() == grad.size());

  const int dim = static_cast<int>(x.rows());
  const double kEps = (std::numeric_limits<double>::epsilon)();

  VectorXd x_p(dim), x_m(dim);

  for (int i = 0; i < dim; ++i) {
    double h = std::pow(3 * kEps, 1.0 / 3.0);
    h = AdjustStepSize(x(i), h);

    x_p = x;
    x_p(i) = x(i) + h;
    x_m = x;
    x_m(i) = x(i) - h;
    grad(i) = (func(x_p) - func(x_m)) / (2 * h);
  }
}

Derivatives NumericalDerivatives(ConvexSet& set, const VectorXd& x,
                                 const VectorXd& x_dot, const VectorXd& dx,
                                 const VectorXd& z, const VectorXd& y) {
  assert(x.rows() == set.nx());
  assert(x_dot.rows() == set.nx());
  assert(dx.rows() == set.ndx());
  assert(z.rows() == set.nz());
  assert(y.rows() == set.nr());

  VectorXd f(set.nr());
  DerivativeFlags flag = DerivativeFlags::f;
  {
    const Derivatives& d = set.UpdateDerivatives(x, dx, z, y, flag);
    f = d.f;
  }

  VectorXd f_x(set.nr());
  for (int i = 0; i < set.nr(); ++i) {
    auto func = [&set, &x, &x_dot, &dx, &z, &y, flag,
                 i](const VectorXd& h_) -> double {
      const Derivatives& d =
          set.UpdateDerivatives(x + x_dot * h_(0), dx, z, y, flag);
      return d.f(i);
    };
    VectorXd grad_(1);
    VectorXd h_(1);
    h_ << 0;
    NumericalGradient(func, h_, grad_);
    f_x(i) = grad_(0);
  }

  MatrixXd f_z(set.nr(), set.nz());
  for (int i = 0; i < set.nr(); ++i) {
    auto func = [&set, &x, &dx, &y, flag, i](const VectorXd& z_) -> double {
      const Derivatives& d = set.UpdateDerivatives(x, dx, z_, y, flag);
      return d.f(i);
    };
    VectorXd grad_(set.nz());
    NumericalGradient(func, z, grad_);
    f_z.block(i, 0, 1, set.nz()) = grad_.transpose();
  }

  VectorXd f_xz_y(set.nz());
  flag = DerivativeFlags::f_x;
  {
    auto func = [&set, &x, &dx, &y, flag](const VectorXd& z_) -> double {
      const Derivatives& d = set.UpdateDerivatives(x, dx, z_, y, flag);
      return y.transpose() * d.f_x;
    };
    NumericalGradient(func, z, f_xz_y);
  }

  MatrixXd f_zz_y(set.nz(), set.nz());
  flag = DerivativeFlags::f_z;
  for (int i = 0; i < set.nz(); ++i) {
    auto func = [&set, &x, &dx, &y, flag, i](const VectorXd& z_) -> double {
      const Derivatives& d = set.UpdateDerivatives(x, dx, z_, y, flag);
      return (y.transpose() * d.f_z)(i);
    };
    VectorXd grad_(set.nz());
    NumericalGradient(func, z, grad_);
    f_zz_y.block(i, 0, 1, set.nz()) = grad_.transpose();
  }

  return Derivatives(f, f_x, f_z, f_xz_y, f_zz_y);
}

void NumericalLieDerivatives(ConvexSet& set, const VectorXd& x,
                             const VectorXd& z, const VectorXd& y,
                             const MatrixXd& fg_dot, const MatrixXd& fg,
                             MatrixXd& L_fg_y) {
  assert(x.rows() == set.nx());
  assert(z.rows() == set.nz());
  assert(y.rows() == set.nr());
  assert(fg_dot.rows() == set.nx());
  assert(fg.rows() == set.ndx());
  assert(fg.cols() == fg_dot.cols());
  assert(L_fg_y.rows() == 1);
  assert(L_fg_y.cols() == fg.cols());

  DerivativeFlags flag = DerivativeFlags::f;
  for (int i = 0; i < fg.cols(); ++i) {
    const auto dx = fg.col(i);
    const auto x_dot = fg_dot.col(i);
    auto func = [&set, &x, &x_dot, &dx, &z, &y,
                 flag](const VectorXd& h_) -> double {
      const Derivatives& d =
          set.UpdateDerivatives(x + x_dot * h_(0), dx, z, y, flag);
      return y.transpose() * d.f;
    };
    VectorXd grad_(1);
    VectorXd h_(1);
    h_ << 0;
    NumericalGradient(func, h_, grad_);
    L_fg_y(i) = grad_(0);
  }
}

}  // namespace sccbf
