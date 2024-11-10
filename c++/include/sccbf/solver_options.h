#ifndef SCCBF_SOLVER_OPTIONS_H_
#define SCCBF_SOLVER_OPTIONS_H_

#include <cassert>
#include <memory>

#include "sccbf/data_types.h"

namespace sccbf {

// LCP options and status
enum class LcpStatus : uint8_t {
  kOptimal,
  kInfeasible,
  kMaxIterReached,
};

struct LcpOptions {
  int max_lcp_iter = 100;
  double ratio_tol = 1e-6;
};

// Full options
struct SolverOptions {
  MatrixXd metric;

  LcpOptions lcp{};

  SolverOptions(int nz);
};

inline SolverOptions::SolverOptions(int nz) {
  metric = MatrixXd::Identity(nz, nz);
}

}  // namespace sccbf

#endif  // SCCBF_SOLVER_OPTIONS_H_
