#include "sccbf/utils.h"

#include <cmath>
#include <cassert>
#include <functional>
#include <limits>

#include "sccbf/data_types.h"

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

}  // namespace sccbf
