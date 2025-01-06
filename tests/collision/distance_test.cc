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
#include "sccbf/solver_options.h"
#include "sccbf/utils/matrix_utils.h"

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

inline void SetRandomState(std::shared_ptr<ConvexSet>& set,
                           const VectorXd& pos) {
  const int nz = set->nz();
  const int nx = set->nx();
  assert((nz == 2 && nx == 6) || (nz == 3 && nx == 12));
  assert(pos.rows() == nz);

  MatrixXd rotation(nz, nz);
  RandomRotation(rotation);
  const VectorXd& x = set->x();
  const VectorXd& dx = set->dx();
  VectorXd x_new(x.rows());
  x_new.head(nz) = pos;
  x_new.tail(nz * nz) = rotation.reshaped(nz * nz, 1);
  set->set_states(x_new, dx);
}

// Assertion function
struct KktError {
  VectorXd dual_inf_err;
  VectorXd prim_inf_err;
  VectorXd compl_err;
  double obj_err;
};

KktError GetKktError(const std::shared_ptr<CollisionPair>& cp) {
  const auto C1 = cp->get_set1();
  const auto C2 = cp->get_set2();
  const int nz = C1->nz() + C2->nz();
  const int nr = C1->nr() + C2->nr();

  // KKT errors.
  VectorXd dual_inf_err(nz);  // gradient condition.
  VectorXd prim_inf_err(nr);
  VectorXd compl_err(nr);
  double obj_err = cp->KktError(dual_inf_err, prim_inf_err, compl_err);
  return KktError{dual_inf_err, prim_inf_err, compl_err, obj_err};
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
  if (kkt_err.compl_err.lpNorm<Eigen::Infinity>() >= tol) {
    success = false;
    failure << "Complementary slackness (inner product) = " << std::endl
            << kkt_err.compl_err.transpose().format(kVecFmt) << std::endl;
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
    solver_ = std::make_shared<DistanceSolver>();
    MatrixXd metric(nz_, nz_);
    const double eps = 1.0;
    RandomSpdMatrix(metric, eps);
    opt_ = std::make_shared<SolverOptions>();
    opt_->metric = metric;

    polytope_ = GetRandomPolytope(nz_);
    ellipsoid_ = GetRandomEllipsoid(nz_);
    nrp_ = polytope_->nr();
    nre_ = ellipsoid_->nr();
  }

  int nz_;
  int nrp_, nre_;
  std::shared_ptr<DistanceSolver> solver_;
  std::shared_ptr<SolverOptions> opt_;
  std::shared_ptr<ConvexSet> polytope_;
  std::shared_ptr<ConvexSet> ellipsoid_;
};

TEST_P(CollisionPairTest, MinimumDistance) {
  VectorXd translation = 10.0 * VectorXd::Ones(nz_) + VectorXd::Random(nz_);

  // Polytope-Ellipsoid distance.
  SetRandomState(polytope_, translation);
  SetRandomState(ellipsoid_, VectorXd::Random(nz_));

  auto cp =
      std::make_shared<CollisionPair>(polytope_, ellipsoid_, opt_, solver_);
  cp->MinimumDistance();
  VectorXd z(2 * nz_), lambda(nrp_ + nre_);
  cp->get_kkt_solution(z, lambda);
  KktError kkt_err = GetKktError(cp);
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
