#ifndef SCCBF_COLLISION_COLLISION_PAIR_H_
#define SCCBF_COLLISION_COLLISION_PAIR_H_

#include <IpTNLP.hpp>
#include <memory>

#include "sccbf/data_types.h"

namespace sccbf {

class ConvexSet;

class CollisionInfo;
class DistanceProblem;
class DistanceSolver;

class CollisionPair {
 public:
  CollisionPair(const std::shared_ptr<ConvexSet>& C1,
                const std::shared_ptr<ConvexSet>& C2,
                const std::shared_ptr<CollisionInfo>& info,
                const std::shared_ptr<DistanceSolver>& solver);

  ~CollisionPair();

  bool MinimumDistance();

  double get_kkt_solution(VectorXd& z, VectorXd& lambda);

  friend class DistanceProblem;

 private:
  void ConvexSetOde_();

  // Convex sets.
  const std::shared_ptr<ConvexSet> C1_;
  const std::shared_ptr<ConvexSet> C2_;
  // Collision information.
  const std::shared_ptr<CollisionInfo> info_;
  // Ipopt minimum distance problem.
  Ipopt::SmartPtr<Ipopt::TNLP> prob_;
  const std::shared_ptr<DistanceSolver> solver_;
  // KKT solution.
  VectorXd z_;
  VectorXd lambda_;
  double dist2_;
};

}  // namespace sccbf

#endif  // SCCBF_COLLISION_COLLISION_PAIR_H_