#include "sccbf/collision/collision_pair.h"

#include <memory>

#include "sccbf/collision/collision_info.h"
#include "sccbf/collision/distance_solver.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

CollisionPair::CollisionPair(const std::shared_ptr<ConvexSet>& C1,
                             const std::shared_ptr<ConvexSet>& C2,
                             const std::shared_ptr<CollisionInfo>& info,
                             const std::shared_ptr<DistanceSolver>& solver)
    : C1_(C1),
      C2_(C2),
      info_(info),
      solver_(solver),
      z_(C1->nz() + C2->nz()),
      lambda_(C1->nr() + C2->nr()),
      dist2_(0.0) {
  assert(C1->dim() == C2->dim());
  assert((info->M.rows() == C1->dim()) && (info->M.cols() == C1->dim()));

  z_ = VectorXd::Zero(C1->nz() + C2->nz());
  lambda_ = VectorXd::Zero(C1->nr() + C2->nr());

  prob_ = new DistanceProblem(*this);
}

CollisionPair::~CollisionPair() {}

bool CollisionPair::MinimumDistance() {
  return solver_->MinimumDistance(prob_);
}

double CollisionPair::get_kkt_solution(VectorXd& z, VectorXd& lambda) {
  assert(z.size() == z_.size());
  assert(lambda.size() == lambda_.size());

  z = z_;
  lambda = lambda_;
  return dist2_;
}

void CollisionPair::ConvexSetOde_() { return; }

}  // namespace sccbf