#ifndef SCCBF_DATA_TYPES_H_
#define SCCBF_DATA_TYPES_H_

#include <Eigen/Core>

namespace sccbf {

template <int rows, int cols>
using Matrixd = Eigen::Matrix<double, rows, cols>;

template <int rows>
using Vectord = Eigen::Matrix<double, rows, 1>;

using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

}  // namespace sccbf

#endif  // SCCBF_DATA_TYPES_H_
