#ifndef SCCBF_UTILS_NUMERICAL_DERIVATIVES_H_
#define SCCBF_UTILS_NUMERICAL_DERIVATIVES_H_

#include <functional>

#include "sccbf/data_types.h"

namespace sccbf {

struct Derivatives;

class ConvexSet;

class DynamicalSystem;

void NumericalGradient(const std::function<double(const VectorXd&)>& func,
                       const VectorXd& x, VectorXd& grad);

Derivatives NumericalDerivatives(ConvexSet& set, const VectorXd& x,
                                 const VectorXd& x_dot, const VectorXd& dx,
                                 const VectorXd& z, const VectorXd& y);

void NumericalLieDerivatives(ConvexSet& set, const VectorXd& x,
                             const VectorXd& z, const VectorXd& y,
                             const MatrixXd& fg_dot, const MatrixXd& fg,
                             MatrixXd& L_fg_y);

void NumericalDynamics(DynamicalSystem& sys, const VectorXd& x, VectorXd& f,
                       MatrixXd& g);

}  // namespace sccbf

#endif  // SCCBF_UTILS_NUMERICAL_DERIVATIVES_H_