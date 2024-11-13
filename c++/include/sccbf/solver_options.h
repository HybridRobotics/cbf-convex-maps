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
  double timestep = 100 * 1e-6;  // [us].

  // Larger values of epsilon values induce smoothness around vertices.
  double index_set_eps = 1e-4;
  double dual_eps = 1e-4;
  // Stability constants.
  double stability_const_dual_inf = 10;
  double stability_const_prim_inf = 50;
  double stability_const_nneg_inf = 10;
  // Small constant for Cholesky decomposition.
  double cholesky_eps = 1e-7;
  // KKT ODE tolerance options.
  bool use_kkt_err_tol = false;
  //  The KKT error metric is unscaled.
  double max_inf_kkt_err = 1e-1;
  //  The primal dual gap is scaled, but is only accurate if the current KKT
  //  solution is primal feasible.
  double max_primal_dual_gap = 10e-3;  // [m].
};

// Full options
struct SolverOptions {
  MatrixXd metric;

  LcpOptions lcp{};

  KktOdeOptions kkt_ode{};
};

}  // namespace sccbf

#endif  // SCCBF_SOLVER_OPTIONS_H_
