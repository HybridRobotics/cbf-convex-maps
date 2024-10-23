#ifndef SCCBF_MATH_UTILS_NUMERICAL_DERIVATIVES_H_
#define SCCBF_MATH_UTILS_NUMERICAL_DERIVATIVES_H_

#include <functional>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

void numerical_gradient(const std::function<double(const VectorXd&)>& func,
                        const VectorXd& x, VectorXd& grad);

Derivatives numerical_derivatives(ConvexSet& set, const VectorXd& x,
                                  const VectorXd& x_dot, const VectorXd& dx,
                                  const VectorXd& z, const VectorXd& y);

void numerical_lie_derivatives_x(ConvexSet& set, const VectorXd& x,
                                 const VectorXd& z, const VectorXd& y,
                                 const MatrixXd& fg_dot, const MatrixXd& fg,
                                 MatrixXd& L_fgA_y);

}  // namespace sccbf

#endif  // SCCBF_MATH_UTILS_NUMERICAL_DERIVATIVES_H_