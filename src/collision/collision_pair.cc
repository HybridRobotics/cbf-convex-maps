#include "sccbf/collision/collision_pair.h"

#include <Eigen/Geometry>
#include <cmath>
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
      nz1_(C1_->nz()),
      nz2_(C2_->nz()),
      nz_(nz1_ + nz2_),
      nr1_(C1_->nr()),
      nr2_(C2_->nr()),
      nr_(nr1_ + nr2_),
      z_opt_(nz_),
      lambda_opt_(nr_),
      hess_cost_(nz_, nz_),
      ldlt_() {
  assert(C1->dim() == C2->dim());
  assert((opt->metric.rows() == C1->dim()) &&
         (opt->metric.cols() == C1->dim()));

  if (!C1->is_strongly_convex() && !C2->is_strongly_convex()) {
    std::runtime_error("Neither convex sets are strongly convex!");
  }

  z_opt_ = VectorXd::Zero(nz_);
  lambda_opt_ = VectorXd::Zero(nr_);
  dist2_opt_ = 0.0;

  const MatrixXd P1 = C1->get_projection_matrix();
  const MatrixXd P2 = C2->get_projection_matrix();
  hess_cost_.topLeftCorner(nz1_, nz1_) = 2 * P1.transpose() * opt->metric * P1;
  hess_cost_.topRightCorner(nz1_, nz2_) =
      -2 * P1.transpose() * opt->metric * P2;
  hess_cost_.bottomLeftCorner(nz2_, nz1_) =
      -2 * P2.transpose() * opt->metric * P1;
  hess_cost_.bottomRightCorner(nz2_, nz2_) =
      2 * P2.transpose() * opt->metric * P2;

  prob_ = new DistanceProblem(*this);
}

CollisionPair::~CollisionPair() {}

bool CollisionPair::MinimumDistance() {
  const bool solved = solver_->MinimumDistance(prob_);
  lambda_opt_ = lambda_opt_.cwiseMax(0.0);
  return solved;
}

double CollisionPair::GetMinimumDistanceDerivative() const {
  const auto z1 = z_opt_.head(nz1_);
  const auto z2 = z_opt_.tail(nz2_);
  const auto lambda1 = lambda_opt_.head(nr1_);
  const auto lambda2 = lambda_opt_.tail(nr2_);

  const DerivativeFlags flag = DerivativeFlags::f_x;
  const Derivatives& d1 = C1_->UpdateDerivatives(z1, lambda1, flag);
  const Derivatives& d2 = C2_->UpdateDerivatives(z2, lambda2, flag);

  return lambda1.dot(d1.f_x) + lambda2.dot(d2.f_x);
}

void CollisionPair::LieDerivatives(const MatrixXd& fg1, const MatrixXd& fg2,
                                   MatrixXd& L_fg_y1, MatrixXd& L_fg_y2) const {
  const auto z1 = z_opt_.head(nz1_);
  const auto z2 = z_opt_.tail(nz2_);
  const auto lambda1 = lambda_opt_.head(nr1_);
  const auto lambda2 = lambda_opt_.tail(nr2_);

  C1_->LieDerivatives(z1, lambda1, fg1, L_fg_y1);
  C2_->LieDerivatives(z2, lambda2, fg2, L_fg_y2);
}

double CollisionPair::KktStep() {
  VectorXd z_t(nz_);
  VectorXd lambda_t(nr_);
  KktOde_(z_t, lambda_t);

  const double dt = opt_->kkt_ode.timestep;
  z_opt_ = z_opt_ + dt * z_t;
  lambda_opt_ = lambda_opt_ + dt * lambda_t;
  lambda_opt_ = lambda_opt_.cwiseMax(0.0);
  dist2_opt_ = 0.5 * z_opt_.transpose() * hess_cost_ * z_opt_;

  double err{};
  double tol{};
  if (opt_->kkt_ode.use_kkt_err_tol) {
    VectorXd dual_inf_err(nz_);
    VectorXd prim_inf_err(nr_);
    VectorXd compl_err(nr_);
    KktError(dual_inf_err, prim_inf_err, compl_err);

    tol = opt_->kkt_ode.max_inf_kkt_err;
    err = std::max({dual_inf_err.lpNorm<Eigen::Infinity>(),
                    prim_inf_err.lpNorm<Eigen::Infinity>(),
                    compl_err.lpNorm<Eigen::Infinity>()});
  } else {
    tol = opt_->kkt_ode.max_primal_dual_gap;
    err = PrimalDualGap_();
  }

  if (err >= tol) {
    MinimumDistance();
  }

  return err;
}

double CollisionPair::KktError(VectorXd& dual_inf_err, VectorXd& prim_inf_err,
                               VectorXd& compl_err) const {
  assert(dual_inf_err.rows() == nz_);
  assert(prim_inf_err.rows() == nr_);
  assert(compl_err.rows() == nr_);

  const auto z1 = z_opt_.head(nz1_);
  const auto z2 = z_opt_.tail(nz2_);
  const auto lambda1 = lambda_opt_.head(nr1_);
  const auto lambda2 = lambda_opt_.tail(nr2_);

  const DerivativeFlags flag = DerivativeFlags::f | DerivativeFlags::f_z;
  const Derivatives& d1 = C1_->UpdateDerivatives(z1, lambda1, flag);
  const Derivatives& d2 = C2_->UpdateDerivatives(z2, lambda2, flag);

  dual_inf_err = hess_cost_ * z_opt_;
  dual_inf_err.head(nz1_) += d1.f_z.transpose() * lambda1;
  dual_inf_err.tail(nz2_) += d2.f_z.transpose() * lambda2;
  prim_inf_err.head(nr1_) = d1.f.cwiseMax(0.0);
  prim_inf_err.tail(nr2_) = d2.f.cwiseMax(0.0);
  compl_err.head(nr1_) = lambda1.cwiseProduct(d1.f);
  compl_err.tail(nr2_) = lambda2.cwiseProduct(d2.f);
  const double obj_err =
      0.5 * z_opt_.transpose() * hess_cost_ * z_opt_ - dist2_opt_;

  return obj_err;
}

const std::shared_ptr<ConvexSet>& CollisionPair::get_set1() { return C1_; }

const std::shared_ptr<ConvexSet>& CollisionPair::get_set2() { return C2_; }

double CollisionPair::get_kkt_solution(VectorXd& z, VectorXd& lambda) const {
  assert(z.size() == z_opt_.size());
  assert(lambda.size() == lambda_opt_.size());

  z = z_opt_;
  lambda = lambda_opt_;
  return dist2_opt_;
}

void CollisionPair::set_kkt_solution(double dist2, VectorXd& z,
                                     VectorXd& lambda) {
  assert(z.size() == z_opt_.size());
  assert(lambda.size() == lambda_opt_.size());

  z_opt_ = z;
  lambda_opt_ = lambda;
  dist2_opt_ = dist2;
}

double CollisionPair::get_minimum_distance() const { return dist2_opt_; }

double CollisionPair::get_margin() const {
  return C1_->get_safety_margin() + C2_->get_safety_margin();
}

double CollisionPair::PrimalDualGap_() {
  const auto z1 = z_opt_.head(nz1_);
  const auto z2 = z_opt_.tail(nz2_);
  const auto lambda1 = lambda_opt_.head(nr1_);
  const auto lambda2 = lambda_opt_.tail(nr2_);

  const DerivativeFlags flag =
      DerivativeFlags::f | DerivativeFlags::f_z | DerivativeFlags::f_zz_y_lb;
  const Derivatives& d1 = C1_->UpdateDerivatives(z1, lambda1, flag);
  const Derivatives& d2 = C2_->UpdateDerivatives(z2, lambda2, flag);

  const double kCholeskyEps = opt_->kkt_ode.cholesky_eps;
  MatrixXd hess_lb = hess_cost_ + kCholeskyEps * MatrixXd::Identity(nz_, nz_);
  hess_lb.topLeftCorner(nz1_, nz1_) += d1.f_zz_y_lb;
  hess_lb.bottomRightCorner(nz2_, nz2_) += d2.f_zz_y_lb;
  ldlt_.compute(hess_lb);

  VectorXd dual_grad = hess_cost_ * z_opt_;
  dual_grad.head(nz1_) += d1.f_z.transpose() * lambda1;
  dual_grad.tail(nz2_) += d2.f_z.transpose() * lambda2;
  const double prim_dual_gap = 0.5 * dual_grad.dot(ldlt_.solve(dual_grad)) -
                               lambda1.dot(d1.f) - lambda2.dot(d2.f);
  return std::max(prim_dual_gap, 0.0);
}

void CollisionPair::KktOde_(VectorXd& z_t, VectorXd& lambda_t) {
  const double kCholeskyEps = opt_->kkt_ode.cholesky_eps;
  const double kKappaDual = opt_->kkt_ode.stability_const_dual_inf;
  const double kKappaPrim = opt_->kkt_ode.stability_const_prim_inf;
  const double kKappaNneg = opt_->kkt_ode.stability_const_nneg_inf;

  const auto z1 = z_opt_.head(nz1_);
  const auto z2 = z_opt_.tail(nz2_);
  const auto lambda1 = lambda_opt_.head(nr1_);
  const auto lambda2 = lambda_opt_.tail(nr2_);

  const DerivativeFlags flag = DerivativeFlags::f | DerivativeFlags::f_x |
                               DerivativeFlags::f_z | DerivativeFlags::f_xz_y |
                               DerivativeFlags::f_zz_y |
                               DerivativeFlags::f_zz_y_lb;
  const Derivatives& d1 = C1_->UpdateDerivatives(z1, lambda1, flag);
  const Derivatives& d2 = C2_->UpdateDerivatives(z2, lambda2, flag);

  VectorXd f(nr_);
  f.head(nr1_) = d1.f;
  f.tail(nr2_) = d2.f;
  VectorXd f_x(nr_);
  f_x.head(nr1_) = d1.f_x;
  f_x.tail(nr2_) = d2.f_x;
  MatrixXd f_z(nr_, nz_);
  f_z.topLeftCorner(nr1_, nz1_) = d1.f_z;
  f_z.topRightCorner(nr1_, nz2_) = MatrixXd::Zero(nr1_, nz2_);
  f_z.bottomLeftCorner(nr2_, nz1_) = MatrixXd::Zero(nr2_, nz1_);
  f_z.bottomRightCorner(nr2_, nz2_) = d2.f_z;
  // Gradient of the Lagrangian.
  const auto L_z = hess_cost_ * z_opt_ + f_z.transpose() * lambda_opt_;
  // Time derivative of the gradient of the Lagrangian.
  VectorXd L_xz(nz_);
  L_xz.head(nz1_) = d1.f_xz_y;
  L_xz.tail(nz2_) = d2.f_xz_y;

  // Index sets.
  // J0c: Inactive primal constraints at z_.
  // J0 : Active primal constraints at z_.
  // J1 : Inactive (non-zero) dual variables.
  // J2e: Active primal constraints at z_, with zero dual.
  // J1c: Active (zero) dual variables.
  // Disjoint union of J0c, J1, and J2e is {1, ..., nr1 + nr2}.
  std::vector<int> J0c, J1, J2e;
  for (int i = 0; i < nr_; ++i) {
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
  const int kkt_mat_size = nz_ + nJ1;
  MatrixXd kkt_mat(kkt_mat_size, kkt_mat_size);
  kkt_mat.topLeftCorner(nz_, nz_) = hess_cost_;
  kkt_mat.topLeftCorner(nz1_, nz1_) += d1.f_zz_y;
  kkt_mat.block(nz1_, nz1_, nz2_, nz2_) += d2.f_zz_y;
  const auto fz_J1 = f_z(J1, Eigen::placeholders::all);
  kkt_mat.bottomLeftCorner(nJ1, nz_) = fz_J1;
  kkt_mat.bottomRightCorner(nJ1, nJ1) = MatrixXd::Zero(nJ1, nJ1);
  kkt_mat += kCholeskyEps * MatrixXd::Identity(kkt_mat_size, kkt_mat_size);
  ldlt_.setZero();
  ldlt_.compute(kkt_mat);
  // RHS vector.
  VectorXd rhs(kkt_mat_size);
  rhs.head(nz_) = -L_xz - kKappaDual * L_z;
  rhs.tail(nJ1) = -f_x(J1) - kKappaPrim * f(J1);

  // Case 1: J2e is the empty set.
  if (J2e.size() == 0) {
    const auto sol = ldlt_.solve(rhs);
    z_t = sol.head(nz_);
    lambda_t(J1) = sol.tail(nJ1);
    lambda_t(J0c) = -kKappaNneg * lambda_opt_(J0c);
  }
  // Case 2: J2e is a non-empty set.
  else {
    const int nJ2e = static_cast<int>(J2e.size());
    const auto fz_J0c = f_z(J0c, Eigen::placeholders::all);
    const auto fz_J2e = f_z(J2e, Eigen::placeholders::all);

    rhs.head(nz_) += kKappaNneg * fz_J0c.transpose() * lambda_opt_(J0c);
    MatrixXd rhs_J2e(kkt_mat_size, nJ2e);
    rhs_J2e.topRows(nz_) = fz_J2e.transpose();
    rhs_J2e.bottomRows(nJ1) = MatrixXd::Zero(nJ1, nJ2e);

    const auto sol = ldlt_.solve(rhs);
    const VectorXd lcp_q = -f_x(J2e) - rhs_J2e.transpose() * sol;
    const auto kkt_inv_rhs_J2e = ldlt_.solve(rhs_J2e);
    const auto temp = rhs_J2e.transpose() * kkt_inv_rhs_J2e;
    const MatrixXd lcp_M = (temp + temp.transpose()) / 2.0;
    VectorXd lcp_sol(nJ2e);
    LcpStatus status = SolveLcp(lcp_M, lcp_q, lcp_sol, opt_->lcp);
    if (status != LcpStatus::kOptimal) {
      lcp_sol = VectorXd::Zero(nJ2e);
    }

    lambda_t(J2e) = lcp_sol;
    lambda_t(J0c) = -kKappaNneg * lambda_opt_(J0c);
    const auto kkt_sol = sol - kkt_inv_rhs_J2e * lcp_sol;
    z_t = kkt_sol.head(nz_);
    lambda_t(J1) = kkt_sol.tail(nJ1);
  }
}

}  // namespace sccbf
