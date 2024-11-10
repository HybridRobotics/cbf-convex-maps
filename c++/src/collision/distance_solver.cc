#include "sccbf/collision/distance_solver.h"

#include <Eigen/Dense>
#include <IpIpoptApplication.hpp>
#include <IpSolveStatistics.hpp>
#include <IpTNLP.hpp>
#include <algorithm>
#include <cassert>
#include <memory>

#include "sccbf/collision/collision_pair.h"
#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/solver_options.h"

namespace sccbf {

DistanceProblem::DistanceProblem(CollisionPair& cp) : cp_(cp) {}

DistanceProblem::~DistanceProblem() {}

// Virtual functions.
bool DistanceProblem::get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                                   Ipopt::Index& nnz_jac_g,
                                   Ipopt::Index& nnz_h_lag,
                                   IndexStyleEnum& index_style) {
  const int nz1 = cp_.C1_->nz();
  const int nz2 = cp_.C2_->nz();
  const int nr1 = cp_.C1_->nr();
  const int nr2 = cp_.C2_->nr();

  n = nz1 + nz2;
  m = nr1 + nr2;
  // Jacobian corresponding to each convex set is dense.
  nnz_jac_g = nz1 * nr1 + nz2 * nr2;
  // Hessian can be dense (in general), but only lower left corner is needed (it
  // is symmetric).
  nnz_h_lag = static_cast<int>((n * n + n) / 2);
  // C style indexing.
  index_style = Ipopt::TNLP::C_STYLE;

  return true;
}

bool DistanceProblem::get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l,
                                      Ipopt::Number* x_u, Ipopt::Index m,
                                      Ipopt::Number* g_l, Ipopt::Number* g_u) {
  std::fill_n(x_l, n, -2e19);
  std::fill_n(x_u, n, 2e19);
  std::fill_n(g_l, m, -2e19);
  std::fill_n(g_u, m, 0.0);

  return true;
}

bool DistanceProblem::get_starting_point(Ipopt::Index n, bool init_x,
                                         Ipopt::Number* x, bool /*init_z*/,
                                         Ipopt::Number* /*z_L*/,
                                         Ipopt::Number* /*z_U*/, Ipopt::Index m,
                                         bool init_lambda,
                                         Ipopt::Number* lambda) {
  if (init_x) {
    for (int i = 0; i < n; ++i) {
      x[i] = cp_.z_(i);
    }
  }
  if (init_lambda) {
    for (int i = 0; i < m; ++i) {
      lambda[i] = cp_.lambda_(i);
    }
  }

  return true;
}

bool DistanceProblem::eval_f(Ipopt::Index /*n*/, const Ipopt::Number* x,
                             bool /*new_x*/, Ipopt::Number& obj_value) {
  const int nz1 = cp_.C1_->nz();
  const int nz2 = cp_.C2_->nz();
  Eigen::Map<const VectorXd> z1(x, nz1);
  Eigen::Map<const VectorXd> z2(x + nz1, nz2);
  const MatrixXd& P1 = cp_.C1_->get_projection_matrix();
  const MatrixXd& P2 = cp_.C2_->get_projection_matrix();
  const auto diff = P1 * z1 - P2 * z2;
  obj_value = diff.transpose() * cp_.opt_->metric * diff;

  return true;
}

bool DistanceProblem::eval_grad_f(Ipopt::Index n, const Ipopt::Number* x,
                                  bool /*new_x*/, Ipopt::Number* grad_f) {
  const int nz1 = cp_.C1_->nz();
  const int nz2 = cp_.C2_->nz();

  Eigen::Map<const VectorXd> z1(x, nz1);
  Eigen::Map<const VectorXd> z2(x + nz1, nz2);
  const MatrixXd& P1 = cp_.C1_->get_projection_matrix();
  const MatrixXd& P2 = cp_.C2_->get_projection_matrix();
  const auto diff = cp_.opt_->metric * (P1 * z1 - P2 * z2);
  const auto grad_z1 = 2 * P1.transpose() * diff;
  const auto grad_z2 = -2 * P2.transpose() * diff;
  for (int i = 0; i < nz1; ++i) {
    grad_f[i] = grad_z1(i);
  }
  for (int i = nz1, j = 0; i < n; ++i, ++j) {
    grad_f[i] = grad_z2(j);
  }

  return true;
}

bool DistanceProblem::eval_g(Ipopt::Index /*n*/, const Ipopt::Number* x,
                             bool /*new_x*/, Ipopt::Index m, Ipopt::Number* g) {
  const int nz1 = cp_.C1_->nz();
  const int nz2 = cp_.C2_->nz();
  const int nr1 = cp_.C1_->nr();
  const int nr2 = cp_.C2_->nr();

  Eigen::Map<const VectorXd> z1(x, nz1);
  Eigen::Map<const VectorXd> z2(x + nz1, nz2);
  const DerivativeFlags flag = DerivativeFlags::f;

  const VectorXd dx1(cp_.C1_->ndx());
  const VectorXd y1(nr1);
  const Derivatives& d1 = cp_.C1_->UpdateDerivatives(z1, y1, flag);
  for (int i = 0; i < nr1; ++i) {
    g[i] = d1.f(i);
  }

  const VectorXd dx2(cp_.C2_->ndx());
  const VectorXd y2(nr2);
  const Derivatives& d2 = cp_.C2_->UpdateDerivatives(z2, y2, flag);
  for (int i = nr1, j = 0; i < m; ++i, ++j) {
    g[i] = d2.f(j);
  }

  return true;
}

bool DistanceProblem::eval_jac_g(Ipopt::Index n, const Ipopt::Number* x,
                                 bool /*new_x*/, Ipopt::Index m,
                                 Ipopt::Index /*nele_jac*/, Ipopt::Index* iRow,
                                 Ipopt::Index* jCol, Ipopt::Number* values) {
  const int nz1 = cp_.C1_->nz();
  const int nz2 = cp_.C2_->nz();
  const int nr1 = cp_.C1_->nr();
  const int nr2 = cp_.C2_->nr();

  if (values == NULL) {
    // Return the structure of the Jacobian.

    // Jacobian has block diagonal sparsity structure.
    // Fill in row and column indices in a column-major order, the same order
    // that Eigen stores its elements by default.
    Ipopt::Index idx = 0;
    for (int col = 0; col < nz1; ++col) {
      for (int row = 0; row < nr1; ++row) {
        iRow[idx] = row;
        jCol[idx] = col;
        idx++;
      }
    }
    for (int col = nz1; col < n; ++col) {
      for (int row = nr1; row < m; ++row) {
        iRow[idx] = row;
        jCol[idx] = col;
        idx++;
      }
    }
  } else {
    // Return the values of the Jacobian of the constraints.

    Eigen::Map<const VectorXd> z1(x, nz1);
    Eigen::Map<const VectorXd> z2(x + nz1, nz2);
    const DerivativeFlags flag = DerivativeFlags::f_z;

    const VectorXd dx1(cp_.C1_->ndx());
    const VectorXd y1(nr1);
    const Derivatives& d1 = cp_.C1_->UpdateDerivatives(z1, y1, flag);
    // Since values are assigned in a column-major order (the same default order
    // as Eigen::MatrixXd), we can directly map the Jacobian elements.
    Eigen::Map<MatrixXd> f1_z(values, nr1, nz1);
    f1_z = d1.f_z;

    const VectorXd dx2(cp_.C2_->ndx());
    const VectorXd y2(nr2);
    const Derivatives& d2 = cp_.C2_->UpdateDerivatives(z2, y2, flag);
    Eigen::Map<MatrixXd> f2_z(values + nr1 * nz1, nr2, nz2);
    f2_z = d2.f_z;
  }

  return true;
}

bool DistanceProblem::eval_h(Ipopt::Index n, const Ipopt::Number* x,
                             bool /*new_x*/, Ipopt::Number obj_factor,
                             Ipopt::Index /*m*/, const Ipopt::Number* lambda,
                             bool /*new_lambda*/, Ipopt::Index /*nele_hess*/,
                             Ipopt::Index* iRow, Ipopt::Index* jCol,
                             Ipopt::Number* values) {
  const int nz1 = cp_.C1_->nz();
  const int nz2 = cp_.C2_->nz();
  const int nr1 = cp_.C1_->nr();
  const int nr2 = cp_.C2_->nr();

  if (values == NULL) {
    // Return the Hessian structure. This is a symmetric matrix, fill the
    // lower-left triangle only.

    Ipopt::Index idx = 0;
    for (int col = 0; col < n; ++col) {
      for (int row = col; row < n; ++row) {
        iRow[idx] = row;
        jCol[idx] = col;
        idx++;
      }
    }
  } else {
    // Return the Hessian values. This is a symmetric matrix, fill the
    // lower-left triangle only.

    Eigen::Map<const VectorXd> z1(x, nz1);
    Eigen::Map<const VectorXd> z2(x + nz1, nz2);
    Eigen::Map<const VectorXd> y1(lambda, nr1);
    Eigen::Map<const VectorXd> y2(lambda + nr1, nr2);
    const DerivativeFlags flag = DerivativeFlags::f_zz_y;

    const VectorXd dx1(cp_.C1_->ndx());
    const Derivatives& d1 = cp_.C1_->UpdateDerivatives(z1, y1, flag);
    const VectorXd dx2(cp_.C2_->ndx());
    const Derivatives& d2 = cp_.C2_->UpdateDerivatives(z2, y2, flag);

    const MatrixXd& P1 = cp_.C1_->get_projection_matrix();
    const MatrixXd& P2 = cp_.C2_->get_projection_matrix();

    MatrixXd hess(n, n);
    const auto hess_11 =
        2 * obj_factor * P1.transpose() * cp_.opt_->metric * P1 + d1.f_zz_y;
    const auto hess_21 =
        -2 * obj_factor * P2.transpose() * cp_.opt_->metric * P1;
    const auto hess_22 =
        2 * obj_factor * P2.transpose() * cp_.opt_->metric * P2 + d2.f_zz_y;
    hess.topLeftCorner(nz1, nz1).triangularView<Eigen::Lower>() =
        hess_11.triangularView<Eigen::Lower>();
    hess.bottomLeftCorner(nz2, nz1) = hess_21;
    hess.bottomRightCorner(nz2, nz2).triangularView<Eigen::Lower>() =
        hess_22.triangularView<Eigen::Lower>();

    Ipopt::Index idx = 0;
    for (int col = 0; col < n; ++col) {
      for (int row = col; row < n; ++row) {
        values[idx] = hess(row, col);
        idx++;
      }
    }
  }

  return true;
}

void DistanceProblem::finalize_solution(
    Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number* x,
    const Ipopt::Number* /*z_L*/, const Ipopt::Number* /*z_U*/, Ipopt::Index m,
    const Ipopt::Number* /*g*/, const Ipopt::Number* lambda,
    Ipopt::Number obj_value, const Ipopt::IpoptData* /*ip_data*/,
    Ipopt::IpoptCalculatedQuantities* /*ip_cq*/
) {
  if ((status == Ipopt::SUCCESS) || (status == Ipopt::MAXITER_EXCEEDED) ||
      (status == Ipopt::CPUTIME_EXCEEDED) ||
      (status == Ipopt::STOP_AT_ACCEPTABLE_POINT)) {
    Eigen::Map<const VectorXd> z_opt(x, n);
    cp_.z_ = z_opt;
    Eigen::Map<const VectorXd> lambda_opt(lambda, m);
    cp_.lambda_ = lambda_opt;
    cp_.dist2_ = obj_value;
  }
}

DistanceSolver::DistanceSolver() {
  app_ = std::make_unique<Ipopt::IpoptApplication>();

  // NLP options.
  SetOption("jacobian_approximation", "exact");
  SetOption("gradient_approximation", "exact");

  // Hessian approximation options.
  SetOption("hessian_approximation", "exact");

  // Warm start options.
  SetOption("warm_start_init_point", "yes");

  // Termination options.                       // [Tune]
  SetOption("max_iter", 1000);
  SetOption("max_wall_time", 1.0);
  SetOption("tol", 1e-4);
  SetOption("dual_inf_tol", 1e-6);
  SetOption("constr_viol_tol", 1e-4);
  SetOption("compl_inf_tol", 1e-8);
  SetOption("acceptable_tol", 1e-3);
  SetOption("acceptable_dual_inf_tol", 1e-5);
  SetOption("acceptable_constr_viol_tol", 1e-3);
  SetOption("compl_inf_tol", 1e-6);

  // Linear solver options.
  // The MA27 solver can be obtained for free (for academic purposes) from
  // http://www.hsl.rl.ac.uk/ipopt/, and must be compiled into the Ipopt
  // library. See Ipopt installation.
  SetOption("linear_solver", "ma27");
  // SetOption("linear_solver", "mumps");

  // Output options.                            // [Debug]
  SetOption("timing_statistics", "no");
  SetOption("print_timing_statistics", "no");
  SetOption("print_user_options", "no");
  SetOption("print_level", 0);  // 0 - 12.
  SetOption("sb", "yes");       // Suppresses copyright information.
  SetOption("print_frequency_time", 1e-5);

  // For debugging derivatives.                 // [Debug]
  SetOption("derivative_test", "none");  // ("none", "second-order")
  SetOption("derivative_test_tol", 1e-3);

#ifndef NDEBUG
  app_->RethrowNonIpoptException(true);
#endif

  Ipopt::ApplicationReturnStatus status = app_->Initialize();
  if (status != Ipopt::Solve_Succeeded) {
    std::runtime_error("*** Error during initialization!");
  }
}

DistanceSolver::~DistanceSolver() {}

bool DistanceSolver::MinimumDistance(Ipopt::SmartPtr<Ipopt::TNLP> prob_ptr) {
  Ipopt::ApplicationReturnStatus status = app_->OptimizeTNLP(prob_ptr);

  return (status == Ipopt::Solve_Succeeded);
}

void DistanceSolver::SetOption(const std::string& name,
                               const std::string& value) {
  app_->Options()->SetStringValue(name, value);
}

void DistanceSolver::SetOption(const std::string& name, int value) {
  app_->Options()->SetIntegerValue(name, value);
}

void DistanceSolver::SetOption(const std::string& name, double value) {
  app_->Options()->SetNumericValue(name, value);
}

}  // namespace sccbf
