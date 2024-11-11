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

// KKT ODE options
struct KktOdeOptions {
  double timestep = 100 * 1e-6; // [us].

  double index_set_eps = 1e-7;
  double dual_eps = 1e-7;
  // Generally, 1 - stability_const * timestep ~= 0.9.
  double stability_const = 100;
  double cholesky_eps = 1e-7;
};

// Full options
struct SolverOptions {
  MatrixXd metric;

  LcpOptions lcp{};

  KktOdeOptions kkt_ode{};
};

}  // namespace sccbf

#endif  // SCCBF_SOLVER_OPTIONS_H_
