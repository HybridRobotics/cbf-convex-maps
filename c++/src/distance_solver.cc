#include "sccbf/distance_solver.h"

#include <Eigen/Dense>
#include <IpIpoptApplication.hpp>
#include <IpSolveStatistics.hpp>
#include <IpTNLP.hpp>
#include <algorithm>
#include <cassert>
#include <memory>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

DistanceProblem::DistanceProblem(const std::shared_ptr<ConvexSet>& C1,
                                 const VectorXd& x1,
                                 const std::shared_ptr<ConvexSet>& C2,
                                 const VectorXd& x2, const MatrixXd& M)
    : C1_(C1),
      C2_(C2),
      x1_(x1),
      x2_(x2),
      M_(M),
      z_(C1_->nz() + C2_->nz()),
      lambda_(C1_->nr() + C2_->nr()) {
  assert(C1->dim() == C2->dim());
  assert(x1.rows() == C1->nx());
  assert(x2.rows() == C2->nx());
  assert((M.rows() == C1->dim()) && (M.cols() == C1->dim()));

  z_ = VectorXd::Zero(C1->nz() + C2->nz());
  lambda_ = VectorXd::Zero(C1->nr() + C2->nr());
}

DistanceProblem::~DistanceProblem() {}

void DistanceProblem::set_states(const VectorXd& x1, const VectorXd& x2) {
  assert(x1.rows() == C1_->nx());
  assert(x2.rows() == C2_->nx());

  x1_ = x1;
  x2_ = x2;
}

// Virtual functions.
bool DistanceProblem::get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                                   Ipopt::Index& nnz_jac_g,
                                   Ipopt::Index& nnz_h_lag,
                                   IndexStyleEnum& index_style) {
  n = C1_->nz() + C2_->nz();
  m = C1_->nr() + C2_->nr();
  // Jacobian corresponding to each convex set is dense.
  nnz_jac_g = C1_->nz() * C1_->nr() + C2_->nz() * C2_->nr();
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
      x[i] = z_(i);
    }
  }
  if (init_lambda) {
    for (int i = 0; i < m; ++i) {
      lambda[i] = lambda_(i);
    }
  }

  return true;
}

bool DistanceProblem::eval_f(Ipopt::Index /*n*/, const Ipopt::Number* x,
                             bool /*new_x*/, Ipopt::Number& obj_value) {
  Eigen::Map<const VectorXd> z1(x, C1_->nz());
  Eigen::Map<const VectorXd> z2(x + C1_->nz(), C2_->nz());
  const MatrixXd& P1 = C1_->get_projection_matrix();
  const MatrixXd& P2 = C2_->get_projection_matrix();
  const auto diff = P1 * z1 - P2 * z2;
  obj_value = diff.transpose() * M_ * diff;

  return true;
}

bool DistanceProblem::eval_grad_f(Ipopt::Index n, const Ipopt::Number* x,
                                  bool /*new_x*/, Ipopt::Number* grad_f) {
  Eigen::Map<const VectorXd> z1(x, C1_->nz());
  Eigen::Map<const VectorXd> z2(x + C1_->nz(), C2_->nz());
  const MatrixXd& P1 = C1_->get_projection_matrix();
  const MatrixXd& P2 = C2_->get_projection_matrix();
  const auto diff = M_ * (P1 * z1 - P2 * z2);
  const auto grad_z1 = 2 * P1.transpose() * diff;
  const auto grad_z2 = -2 * P2.transpose() * diff;
  for (int i = 0; i < C1_->nz(); ++i) {
    grad_f[i] = grad_z1(i);
  }
  for (int i = C1_->nz(), j = 0; i < n; ++i, ++j) {
    grad_f[i] = grad_z2(j);
  }

  return true;
}

bool DistanceProblem::eval_g(Ipopt::Index /*n*/, const Ipopt::Number* x,
                             bool /*new_x*/, Ipopt::Index m, Ipopt::Number* g) {
  Eigen::Map<const VectorXd> z1(x, C1_->nz());
  Eigen::Map<const VectorXd> z2(x + C1_->nz(), C2_->nz());
  const DerivativeFlags flag = DerivativeFlags::f;

  const VectorXd dx1(C1_->ndx());
  const VectorXd y1(C1_->nr());
  const Derivatives& d1 = C1_->UpdateDerivatives(x1_, dx1, z1, y1, flag);
  for (int i = 0; i < C1_->nr(); ++i) {
    g[i] = d1.f(i);
  }

  const VectorXd dx2(C2_->ndx());
  const VectorXd y2(C2_->nr());
  const Derivatives& d2 = C2_->UpdateDerivatives(x2_, dx2, z2, y2, flag);
  for (int i = C1_->nr(), j = 0; i < m; ++i, ++j) {
    g[i] = d2.f(j);
  }

  return true;
}

bool DistanceProblem::eval_jac_g(Ipopt::Index n, const Ipopt::Number* x,
                                 bool /*new_x*/, Ipopt::Index m,
                                 Ipopt::Index /*nele_jac*/, Ipopt::Index* iRow,
                                 Ipopt::Index* jCol, Ipopt::Number* values) {
  if (values == NULL) {
    // Return the structure of the Jacobian.

    // Jacobian has block diagonal sparsity structure.
    // Fill in row and column indices in a column-major order, the same order
    // that Eigen stores its elements by default.
    Ipopt::Index idx = 0;
    for (int col = 0; col < C1_->nz(); ++col) {
      for (int row = 0; row < C1_->nr(); ++row) {
        iRow[idx] = row;
        jCol[idx] = col;
        idx++;
      }
    }
    for (int col = C1_->nz(); col < n; ++col) {
      for (int row = C1_->nr(); row < m; ++row) {
        iRow[idx] = row;
        jCol[idx] = col;
        idx++;
      }
    }
  } else {
    // Return the values of the Jacobian of the constraints.

    Eigen::Map<const VectorXd> z1(x, C1_->nz());
    Eigen::Map<const VectorXd> z2(x + C1_->nz(), C2_->nz());
    const DerivativeFlags flag = DerivativeFlags::f_z;

    const VectorXd dx1(C1_->ndx());
    const VectorXd y1(C1_->nr());
    const Derivatives& d1 = C1_->UpdateDerivatives(x1_, dx1, z1, y1, flag);
    // Since values are assigned in a column-major order (the same default order
    // as Eigen::MatrixXd), we can directly map the Jacobian elements.
    Eigen::Map<MatrixXd> f1_z(values, C1_->nr(), C1_->nz());
    f1_z = d1.f_z;

    const VectorXd dx2(C2_->ndx());
    const VectorXd y2(C2_->nr());
    const Derivatives& d2 = C2_->UpdateDerivatives(x2_, dx2, z2, y2, flag);
    Eigen::Map<MatrixXd> f2_z(values + C1_->nr() * C1_->nz(), C2_->nr(),
                              C2_->nz());
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

    Eigen::Map<const VectorXd> z1(x, C1_->nz());
    Eigen::Map<const VectorXd> z2(x + C1_->nz(), C2_->nz());
    Eigen::Map<const VectorXd> y1(lambda, C1_->nr());
    Eigen::Map<const VectorXd> y2(lambda + C1_->nr(), C2_->nr());
    const DerivativeFlags flag = DerivativeFlags::f_zz_y;

    const VectorXd dx1(C1_->ndx());
    const Derivatives& d1 = C1_->UpdateDerivatives(x1_, dx1, z1, y1, flag);
    const VectorXd dx2(C2_->ndx());
    const Derivatives& d2 = C2_->UpdateDerivatives(x2_, dx2, z2, y2, flag);

    const MatrixXd& P1 = C1_->get_projection_matrix();
    const MatrixXd& P2 = C2_->get_projection_matrix();

    MatrixXd hess(n, n);
    const auto hess_11 = 2 * obj_factor * P1.transpose() * M_ * P1 + d1.f_zz_y;
    const auto hess_21 = -2 * obj_factor * P2.transpose() * M_ * P1;
    const auto hess_22 = 2 * obj_factor * P2.transpose() * M_ * P2 + d2.f_zz_y;
    hess.topLeftCorner(C1_->nz(), C1_->nz()).triangularView<Eigen::Lower>() =
        hess_11.triangularView<Eigen::Lower>();
    hess.bottomLeftCorner(C2_->nz(), C1_->nz()) = hess_21;
    hess.bottomRightCorner(C2_->nz(), C2_->nz())
        .triangularView<Eigen::Lower>() =
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
    z_ = z_opt;
    Eigen::Map<const VectorXd> lambda_opt(lambda, m);
    lambda_ = lambda_opt;
    dist2_ = obj_value;
  }
}

DistanceSolver::DistanceSolver() {
  app_ = std::make_unique<Ipopt::IpoptApplication>();

  // The MA27 solver can be obtained for free (for academic purposes) from
  // http://www.hsl.rl.ac.uk/ipopt/, and must be compiled into the Ipopt
  // library. See Ipopt installation. SetOption("linear_solver", "mumps");
  SetOption("linear_solver", "ma27");

  // Print levels.
  SetOption("print_timing_statistics", "no");
  SetOption("print_user_options", "no");
  SetOption("print_level", 4);

  // For debugging derivatives.
  SetOption("derivative_test", "second-order");
  SetOption("derivative_test_tol", 1e-3);

#ifndef NDEBUG
  app_->RethrowNonIpoptException(true);
#endif
}

DistanceSolver::~DistanceSolver() {}

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

double DistanceSolver::GetTotalWallclockTime() {
  return app_->Statistics()->TotalWallclockTime();
}

}  // namespace sccbf
