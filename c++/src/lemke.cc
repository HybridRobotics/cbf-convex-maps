#define SCCBF_ROBUST_LEMKE

#include "sccbf/lemke.h"

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <limits>

#include "sccbf/data_types.h"
#include "sccbf/solver_options.h"

namespace sccbf {

namespace {

inline void PivotColumn(MatrixXd& tableau, int row, int col, int n) {
  tableau.row(row) /= tableau(row, col);
  for (int i = 0; i < n; ++i) {
    if (i != row) {
      tableau.row(i) -= tableau.row(row) * tableau(i, col);
    }
  }
}

inline int LexicoMinimumIndex(const MatrixXd& tableau, int idx1, int idx2,
                              int entering_idx, int n) {
  for (int i = 0; i < n; ++i) {
    const double diff = tableau(idx2, i) * tableau(idx1, entering_idx) -
                        tableau(idx1, i) * tableau(idx2, entering_idx);
    if (diff < 0)
      return idx2;
    else if (diff > 0)
      return idx1;
  }
  return idx1;
}

}  // namespace

LcpStatus SolveLcp(const MatrixXd& M, const VectorXd& q, VectorXd& z,
                   const LcpOptions& opt) {
  const int n = static_cast<int>(M.rows());
  assert(M.cols() == n);
  assert(q.rows() == n);
  assert(z.rows() == n);

  MatrixXd tableau(n, 2 * n + 2);
  tableau.block(0, 0, n, n) = MatrixXd::Identity(n, n);
  tableau.block(0, n, n, n) = -M;
  tableau.col(2 * n) = -VectorXd::Ones(n);
  tableau.col(2 * n + 1) = q;
  Eigen::VectorXi basic_vars = Eigen::VectorXi::LinSpaced(n, 0, n - 1);

  int entering_idx{2 * n}, exiting_idx{};
  const int q_idx = 2 * n + 1;
  double ratio{}, min_ratio{};

  // Initialization.
  min_ratio = tableau.col(q_idx).minCoeff(&exiting_idx);
  if (min_ratio > -opt.ratio_tol) {
    z = Eigen::VectorXd::Zero(n);
    return LcpStatus::kOptimal;
  }
  PivotColumn(tableau, exiting_idx, entering_idx, n);
  tableau.col(q_idx) = tableau.col(q_idx).cwiseMax(0.0);
  const int z0_idx{exiting_idx};
  basic_vars(exiting_idx) = entering_idx;
  entering_idx = exiting_idx + n;

  for (int iter = 0; iter < opt.max_lcp_iter; ++iter) {
    // Find exiting index.
    min_ratio = std::numeric_limits<double>::infinity();
    exiting_idx = -1;
    for (int i = 0; i < n; ++i) {
      if (tableau(i, entering_idx) > 0) {
        ratio = tableau(i, q_idx) / tableau(i, entering_idx);
        if (ratio < min_ratio) {
          min_ratio = ratio;
          exiting_idx = i;
        }
      }
    }
    if (exiting_idx == -1) {
      return LcpStatus::kInfeasible;
    }

#ifdef SCCBF_ROBUST_LEMKE
    int exiting_idx_{exiting_idx};
    for (int i = 0; i < n; ++i) {
      if ((i != exiting_idx) && (tableau(i, entering_idx) > 0)) {
        if (std::abs(tableau(i, q_idx) / tableau(i, entering_idx) - min_ratio) <
            opt.ratio_tol) {
          if (i == z0_idx) {
            exiting_idx_ = z0_idx;
            break;
          }
          exiting_idx_ =
              LexicoMinimumIndex(tableau, exiting_idx_, i, entering_idx, n);
        }
      }
    }
    exiting_idx = exiting_idx_;
#endif  // SCCBF_ROBUST_LEMKE

    // Pivot and compute entering index.
    PivotColumn(tableau, exiting_idx, entering_idx, n);
    tableau.col(q_idx) = tableau.col(q_idx).cwiseMax(0.0);
    const int exiting_var = basic_vars(exiting_idx);
    basic_vars(exiting_idx) = entering_idx;
    if (exiting_var < n) {
      entering_idx = exiting_var + n;
    } else if (exiting_var < 2 * n) {
      entering_idx = exiting_var - n;
    } else {
      z = VectorXd::Zero(n);
      for (int i = 0; i < n; ++i) {
        if (basic_vars(i) >= n) {
          z(basic_vars(i) - n) = tableau(i, q_idx);
        }
      }
      return LcpStatus::kOptimal;
    }
  }
  return LcpStatus::kMaxIterReached;
}

}  // namespace sccbf
