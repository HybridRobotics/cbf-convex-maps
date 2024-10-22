#ifndef SCCBF_UTILS_H_
#define SCCBF_UTILS_H_

#include <Eigen/Core>
#include <cassert>
#include <cmath>

#include "sccbf/data_types.h"

namespace sccbf {

// res should be set to zero along the diagonal.
template <int dim, typename Derived>
inline void hat_map(const Eigen::MatrixBase<Derived>& vec, MatrixXd& res) {
  static_assert((dim == 2) || (dim == 3));
  assert(vec.rows() == ((dim == 2) ? 1 : 3));
  assert(vec.cols() == 1);
  assert((res.rows() == dim) && (res.cols() == dim));

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

}  // namespace sccbf

#endif  // SCCBF_UTILS_H_
