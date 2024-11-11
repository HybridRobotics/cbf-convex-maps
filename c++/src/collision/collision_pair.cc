#include "sccbf/collision/collision_pair.h"

#include <Eigen/Geometry>
#include <memory>
#include <vector>

#include "sccbf/collision/distance_solver.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/lemke.h"
#include "sccbf/solver_options.h"

namespace sccbf {

CollisionPair::CollisionPair(const std::shared_ptr<ConvexSet>& C1,
                             const std::shared_ptr<ConvexSet>& C2,
                             const std::shared_ptr<SolverOptions>& opt,
                             const std::shared_ptr<DistanceSolver>& solver)
    : C1_(C1),
      C2_(C2),
      opt_(opt),
      solver_(solver),
      z_opt_(C1->nz() + C2->nz()),
      lambda_opt_(C1->nr() + C2->nr()),
      dist2_opt_(0.0),
      hess_cost_(C1->nz() + C2->nz(), C1->nz() + C2->nz()),
      kkt_ldlt_() {
  assert(C1->dim() == C2->dim());
  assert((opt->metric.rows() == C1->dim()) &&
         (opt->metric.cols() == C1->dim()));

  if (!C1->is_strongly_convex() && !C2->is_strongly_convex()) {
    std::runtime_error("Neither convex sets are strongly convex!");
  }

  const int nz1 = C1->nz();
  const int nz2 = C2->nz();

  z_opt_.head(nz1) = C1->get_center();
  z_opt_.tail(nz2) = C2->get_center();
  lambda_opt_ = VectorXd::Zero(C1->nr() + C2->nr());

  const MatrixXd P1 = C1->get_projection_matrix();
  const MatrixXd P2 = C2->get_projection_matrix();
  hess_cost_.topLeftCorner(nz1, nz1) = 2 * P1.transpose() * opt->metric * P1;
  hess_cost_.topRightCorner(nz1, nz2) = -2 * P1.transpose() * opt->metric * P2;
  hess_cost_.bottomLeftCorner(nz2, nz1) =
      -2 * P2.transpose() * opt->metric * P1;
  hess_cost_.bottomRightCorner(nz2, nz2) =
      2 * P2.transpose() * opt->metric * P2;

  prob_ = new DistanceProblem(*this);
}

CollisionPair::~CollisionPair() {}

bool CollisionPair::MinimumDistance() {
  return solver_->MinimumDistance(prob_);
}

double CollisionPair::KktError(VectorXd& dual_inf_err, VectorXd& prim_inf_err,
                               VectorXd& compl_err) const {
  const int nz1 = C1_->nz();
  const int nz2 = C2_->nz();
  const int nz = nz1 + nz2;
  const int nr1 = C1_->nr();
  const int nr2 = C2_->nr();
  const int nr = nr1 + nr2;

  assert(dual_inf_err.rows() == nz);
  assert(prim_inf_err.rows() == nr);
  assert(compl_err.rows() == nr);

  const auto z1 = z_opt_.head(nz1);
  const auto z2 = z_opt_.tail(nz2);
  const auto lambda1 = lambda_opt_.head(nr1);
  const auto lambda2 = lambda_opt_.tail(nr2);

  const DerivativeFlags flag =
      DerivativeFlags::f | DerivativeFlags::f_z | DerivativeFlags::f_zz_y;
  const Derivatives& d1 = C1_->UpdateDerivatives(z1, lambda1, flag);
  const Derivatives& d2 = C2_->UpdateDerivatives(z2, lambda2, flag);

  dual_inf_err = hess_cost_ * z_opt_;
  dual_inf_err.head(nz1) += d1.f_z.transpose() * lambda1;
  dual_inf_err.tail(nz2) += d2.f_z.transpose() * lambda2;
  prim_inf_err.head(nr1) = d1.f.cwiseMax(0.0);
  prim_inf_err.tail(nr2) = d2.f.cwiseMax(0.0);
  compl_err.head(nr1) = lambda1.cwiseProduct(d1.f);
  compl_err.tail(nr2) = lambda2.cwiseProduct(d2.f);
  const double obj_err =
      0.5 * z_opt_.transpose() * hess_cost_ * z_opt_ - dist2_opt_;

  return obj_err;
}

const std::shared_ptr<ConvexSet>& CollisionPair::get_set1() { return C1_; }

const std::shared_ptr<ConvexSet>& CollisionPair::get_set2() { return C2_; }

double CollisionPair::get_kkt_solution(VectorXd& z_opt,
                                       VectorXd& lambda_opt) const {
  assert(z_opt.size() == z_opt_.size());
  assert(lambda_opt.size() == lambda_opt_.size());

  z_opt = z_opt_;
  lambda_opt = lambda_opt_;
  return dist2_opt_;
}

void CollisionPair::KktOde_(VectorXd& z_t, VectorXd& lambda_t) {
  const double kCholeskyEps = opt_->kkt_ode.cholesky_eps;
  const double kKappa = opt_->kkt_ode.stability_const;

  const int nz1 = C1_->nz();
  const int nz2 = C2_->nz();
  const int nz = nz1 + nz2;
  const int nr1 = C1_->nr();
  const int nr2 = C2_->nr();
  const int nr = nr1 + nr2;

  const auto z1 = z_opt_.head(nz1);
  const auto z2 = z_opt_.tail(nz2);
  const auto lambda1 = lambda_opt_.head(nr1);
  const auto lambda2 = lambda_opt_.tail(nr2);

  const DerivativeFlags flag = DerivativeFlags::f | DerivativeFlags::f_x |
                               DerivativeFlags::f_z | DerivativeFlags::f_xz_y |
                               DerivativeFlags::f_zz_y;
  const Derivatives& d1 = C1_->UpdateDerivatives(z1, lambda1, flag);
  const Derivatives& d2 = C2_->UpdateDerivatives(z2, lambda2, flag);

  VectorXd f(nr);
  f.head(nr1) = d1.f;
  f.tail(nr2) = d2.f;
  VectorXd f_x(nr);
  f_x.head(nr1) = d1.f_x;
  f_x.tail(nr2) = d2.f_x;
  MatrixXd f_z(nr, nz);
  f_z.topLeftCorner(nr1, nz1) = d1.f_z;
  f_z.topRightCorner(nr1, nz2) = MatrixXd::Zero(nr1, nz2);
  f_z.bottomLeftCorner(nr2, nz1) = MatrixXd::Zero(nr2, nz1);
  f_z.bottomRightCorner(nr2, nz2) = d2.f_z;
  // Gradient of the Lagrangian.
  const auto L_z = hess_cost_ * z_opt_ + f_z.transpose() * lambda_opt_;
  // Time derivative of the gradient of the Lagrangian.
  VectorXd L_xz(nz);
  L_xz.head(nz1) = d1.f_xz_y;
  L_xz.tail(nz2) = d2.f_xz_y;

  // Index sets.
  // J0c: Inactive primal constraints at z_.
  // J0 : Active primal constraints at z_.
  // J1 : Inactive (non-zero) dual variables.
  // J2e: Active primal constraints at z_, with zero dual.
  // J1c: Active (zero) dual variables.
  // Disjoint union of J0c, J1, and J2e is {1, ..., nr1 + nr2}.
  std::vector<int> J0c, J1, J2e;
  for (int i = 0; i < nr; ++i) {
    if (f(i) < -opt_->kkt_ode.index_set_eps) {
      J0c.push_back(i);
    } else if (lambda_opt_(i) > opt_->kkt_ode.dual_eps) {
      J1.push_back(i);
    } else {
      J2e.push_back(i);
    }
  }

  // KKT Matrix.
  const int nJ1 = static_cast<int>(J1.size());
  const int kkt_mat_size = nz + nJ1;
  MatrixXd kkt_mat(kkt_mat_size, kkt_mat_size);
  kkt_mat.topLeftCorner(nz, nz) = hess_cost_;
  kkt_mat.topLeftCorner(nz1, nz1) += d1.f_zz_y;
  kkt_mat.block(nz1, nz1, nz2, nz2) += d2.f_zz_y;
  const auto fz_J1 = f_z(J1, Eigen::placeholders::all);
  kkt_mat.bottomLeftCorner(nJ1, nz) = fz_J1;
  kkt_mat.bottomRightCorner(nJ1, nJ1) = MatrixXd::Zero(nJ1, nJ1);
  kkt_mat += kCholeskyEps * MatrixXd::Identity(kkt_mat_size, kkt_mat_size);
  kkt_ldlt_.setZero();
  kkt_ldlt_.compute(kkt_mat);
  // RHS vector.
  VectorXd rhs(kkt_mat_size);
  rhs.head(nz) = -L_xz - kKappa * L_z;
  rhs.tail(nJ1) = -f_x(J1) - kKappa * f(J1);

  // Case 1: J2e is the empty set.
  if (J2e.size() == 0) {
    const auto sol = kkt_ldlt_.solve(rhs);
    z_t = sol.head(nz);
    lambda_t(J1) = sol.tail(nJ1);
    lambda_t(J0c) = -kKappa * lambda_opt_(J0c);
  }
  // Case 2: J2e is a non-empty set.
  else {
    const int nJ2e = static_cast<int>(J2e.size());
    const auto fz_J0c = f_z(J0c, Eigen::placeholders::all);
    const auto fz_J2e = f_z(J2e, Eigen::placeholders::all);

    rhs.head(nz) += kKappa * fz_J0c.transpose() * lambda_opt_(J0c);
    MatrixXd rhs_J2e(kkt_mat_size, nJ2e);
    rhs_J2e.topRows(nz) = fz_J2e.transpose();
    rhs_J2e.bottomRows(nJ1) = MatrixXd::Zero(nJ1, nJ2e);

    const auto sol = kkt_ldlt_.solve(rhs);
    const VectorXd lcp_q = -f_x(J2e) - rhs_J2e.transpose() * sol;
    const auto kkt_inv_rhs_J2e = kkt_ldlt_.solve(rhs_J2e);
    const auto temp = rhs_J2e.transpose() * kkt_inv_rhs_J2e;
    const MatrixXd lcp_M = (temp + temp.transpose()) / 2.0;
    VectorXd lcp_sol(nJ2e);
    LcpStatus status = SolveLcp(lcp_M, lcp_q, lcp_sol, opt_->lcp);
    if (status != LcpStatus::kOptimal) {
      lcp_sol = VectorXd::Zero(nJ2e);
    }

    lambda_t(J2e) = lcp_sol;
    lambda_t(J0c) = -kKappa * lambda_opt_(J0c);
    const auto kkt_sol = sol - kkt_inv_rhs_J2e * lcp_sol;
    z_t = kkt_sol.head(nz);
    lambda_t(J1) = kkt_sol.tail(nJ1);
  }
}

}  // namespace sccbf