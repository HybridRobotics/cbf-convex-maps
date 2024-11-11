#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <string>

#include "sccbf/collision/collision_pair.h"
#include "sccbf/collision/distance_solver.h"
#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/geometry/ellipsoid.h"
#include "sccbf/geometry/polytope.h"
#include "sccbf/math_utils/utils.h"
#include "sccbf/solver_options.h"

namespace {

using namespace sccbf;

std::shared_ptr<ConvexSet> GetRandomPolytope(int nz) {
  assert((nz == 2) || (nz == 3));

  const int nr = 5 * nz;
  const VectorXd center = VectorXd::Zero(nz);
  MatrixXd A(nr, nz);
  VectorXd b(nr);
  const double in_radius = nz * 1.0;
  RandomPolytope(center, in_radius, A, b);
  const double sc_modulus = 1e-2;
  if (nz == 2)
    return std::make_shared<Polytope<2>>(A, b, 0.0, sc_modulus, true);
  else
    return std::make_shared<Polytope<3>>(A, b, 0.0, sc_modulus, true);
}

std::shared_ptr<ConvexSet> GetRandomEllipsoid(int nz) {
  assert((nz == 2) || (nz == 3));

  MatrixXd Q(nz, nz);
  const double eps = 5e-1;
  RandomSpdMatrix(Q, eps);
  const double margin = 0;
  if (nz == 2)
    return std::make_shared<Ellipsoid<2>>(Q, margin);
  else
    return std::make_shared<Ellipsoid<3>>(Q, margin);
}

inline void SetRandomState(std::shared_ptr<ConvexSet>& set_ptr,
                           const VectorXd& pos) {
  const int nz = set_ptr->nz();
  const int nx = set_ptr->nx();
  assert((nz == 2 && nx == 6) || (nz == 3 && nx == 12));
  assert(pos.rows() == nz);

  MatrixXd rotation(nz, nz);
  RandomRotation(rotation);
  const VectorXd& x = set_ptr->x();
  const VectorXd& dx = set_ptr->dx();
  VectorXd x_new(x.rows());
  x_new.head(nz) = pos;
  x_new.tail(nz * nz) = rotation.reshaped(nz * nz, 1);
  set_ptr->set_states(x_new, dx);
}

// Assertion function
const DerivativeFlags kFlag = DerivativeFlags::f | DerivativeFlags::f_z;

struct KktError {
  VectorXd dual_inf_err;
  VectorXd prim_inf_err;
  VectorXd dual_nonneg_err;
  VectorXd compl_slack_err;
  double obj_err;
};

KktError GetKktError(const std::shared_ptr<ConvexSet>& set_ptr1,
                     const std::shared_ptr<ConvexSet>& set_ptr2,
                     const std::shared_ptr<SolverOptions>& opt_ptr,
                     const VectorXd& z, const VectorXd& lambda, double dist2) {
  const int dim = set_ptr1->dim();
  assert(set_ptr2->dim() == dim);
  const int nz1 = set_ptr1->nz();
  const int nz2 = set_ptr2->nz();
  const int nr1 = set_ptr1->nr();
  const int nr2 = set_ptr2->nr();
  VectorXd z1 = z.head(nz1);
  VectorXd z2 = z.tail(nz2);
  VectorXd lambda1 = lambda.head(nr1);
  VectorXd lambda2 = lambda.tail(nr2);
  MatrixXd metric = opt_ptr->metric;
  MatrixXd P1 = set_ptr1->get_projection_matrix();
  MatrixXd P2 = set_ptr2->get_projection_matrix();

  // Update derivatives.
  const Derivatives& d1 = set_ptr1->UpdateDerivatives(z1, lambda1, kFlag);
  const Derivatives& d2 = set_ptr2->UpdateDerivatives(z2, lambda2, kFlag);

  // KKT errors.
  VectorXd dual_inf_err(2 * dim);  // gradient condition.
  dual_inf_err.head(dim) = 2 * P1.transpose() * metric * (P1 * z1 - P2 * z2) +
                           d1.f_z.transpose() * lambda1;
  dual_inf_err.tail(dim) = 2 * P2.transpose() * metric * (P2 * z2 - P1 * z1) +
                           d2.f_z.transpose() * lambda2;
  VectorXd prim_inf_err(nr1 + nr2);
  prim_inf_err.head(nr1) = d1.f.cwiseMax(0.0);
  prim_inf_err.tail(nr2) = d2.f.cwiseMax(0.0);
  VectorXd dual_nonneg_err = (-lambda).cwiseMax(0.0);
  VectorXd compl_slack_err(nr1 + nr2);
  compl_slack_err.head(nr1) = lambda1.cwiseProduct(d1.f);
  compl_slack_err.tail(nr2) = lambda2.cwiseProduct(d2.f);
  double obj_err =
      (P1 * z1 - P2 * z2).transpose() * metric * (P1 * z1 - P2 * z2) - dist2;
  return KktError{dual_inf_err, prim_inf_err, dual_nonneg_err, compl_slack_err,
                  obj_err};
}

Eigen::IOFormat kVecFmt(4, Eigen::DontAlignCols, ", ", "\n", "[", "]");

testing::AssertionResult AssertKktError(const char* /*kkt_err_expr*/,
                                        const char* /*z_expr*/,
                                        const char* /*lambda_expr*/,
                                        const char* /*tol_expr*/,
                                        const KktError& kkt_err,
                                        const VectorXd& z,
                                        const VectorXd& lambda, double tol) {
  auto failure = testing::AssertionFailure();

  bool success = true;

  if (kkt_err.dual_inf_err.norm() >= tol) {
    success = false;
    failure << "Dual infeasibility error = " << std::endl
            << kkt_err.dual_inf_err.transpose().format(kVecFmt) << std::endl;
  }
  if (kkt_err.prim_inf_err.lpNorm<Eigen::Infinity>() >= tol) {
    success = false;
    failure << "Primal infeasibility error = " << std::endl
            << kkt_err.prim_inf_err.transpose().format(kVecFmt) << std::endl;
  }
  if (kkt_err.dual_nonneg_err.lpNorm<Eigen::Infinity>() >= tol) {
    success = false;
    failure << "Dual vector = " << std::endl
            << lambda.transpose().format(kVecFmt) << std::endl;
  }
  if (kkt_err.compl_slack_err.lpNorm<Eigen::Infinity>() >= tol) {
    success = false;
    failure << "Complementary slackness (inner product) = " << std::endl
            << kkt_err.compl_slack_err.transpose().format(kVecFmt) << std::endl;
  }
  if (std::abs(kkt_err.obj_err) >= tol) {
    success = false;
    failure << "Objective mismatch error = " << kkt_err.obj_err << std::endl;
  }

  if (success) return testing::AssertionSuccess();

  failure << "Failure at z = " << std::endl
          << z.transpose().format(kVecFmt) << std::endl
          << ", lambda = " << std::endl
          << lambda.transpose().format(kVecFmt);

  return failure;
}

// Value-parametrized test (dimension).
class CollisionPairTest : public testing::TestWithParam<int> {
 protected:
  CollisionPairTest() {
    nz_ = GetParam();
    solver_ptr_ = std::make_shared<DistanceSolver>();
    MatrixXd metric(nz_, nz_);
    const double eps = 1.0;
    RandomSpdMatrix(metric, eps);
    opt_ptr_ = std::make_shared<SolverOptions>();
    opt_ptr_->metric = metric;

    polytope_ptr_ = GetRandomPolytope(nz_);
    ellipsoid_ptr_ = GetRandomEllipsoid(nz_);
    nrp_ = polytope_ptr_->nr();
    nre_ = ellipsoid_ptr_->nr();
  }

  int nz_;
  int nrp_, nre_;
  std::shared_ptr<DistanceSolver> solver_ptr_;
  std::shared_ptr<SolverOptions> opt_ptr_;
  std::shared_ptr<ConvexSet> polytope_ptr_;
  std::shared_ptr<ConvexSet> ellipsoid_ptr_;
};

TEST_P(CollisionPairTest, DistanceOptimization) {
  VectorXd translation = 10.0 * VectorXd::Ones(nz_) + VectorXd::Random(nz_);

  // Polytope-Ellipsoid distance.
  SetRandomState(polytope_ptr_, translation);
  SetRandomState(ellipsoid_ptr_, VectorXd::Random(nz_));

  auto cp = CollisionPair(polytope_ptr_, ellipsoid_ptr_, opt_ptr_, solver_ptr_);
  cp.MinimumDistance();
  VectorXd z(2 * nz_), lambda(nrp_ + nre_);
  double dist2 = cp.get_kkt_solution(z, lambda);
  KktError kkt_err =
      GetKktError(polytope_ptr_, ellipsoid_ptr_, opt_ptr_, z, lambda, dist2);
  EXPECT_PRED_FORMAT4(AssertKktError, kkt_err, z, lambda, 1e-5);
}

INSTANTIATE_TEST_SUITE_P(
    , CollisionPairTest, testing::Values(2, 3),
    [](const testing::TestParamInfo<CollisionPairTest::ParamType>& info)
        -> std::string {
      std::string name = std::to_string(info.param) + "D";
      return name;
    });

}  // namespace
