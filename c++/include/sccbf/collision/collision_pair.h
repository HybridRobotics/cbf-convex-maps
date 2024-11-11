#ifndef SCCBF_COLLISION_COLLISION_PAIR_H_
#define SCCBF_COLLISION_COLLISION_PAIR_H_

#include <Eigen/Geometry>
#include <IpTNLP.hpp>
#include <memory>

#include "sccbf/data_types.h"

namespace sccbf {

class ConvexSet;

class SolverOptions;
class DistanceProblem;
class DistanceSolver;

class CollisionPair {
 public:
  CollisionPair(const std::shared_ptr<ConvexSet>& C1,
                const std::shared_ptr<ConvexSet>& C2,
                const std::shared_ptr<SolverOptions>& opt,
                const std::shared_ptr<DistanceSolver>& solver);

  ~CollisionPair();

  bool MinimumDistance();

  const std::shared_ptr<ConvexSet>& get_set1();

  const std::shared_ptr<ConvexSet>& get_set2();

  double KktError(VectorXd& dual_inf_err, VectorXd& prim_inf_err,
                  VectorXd& compl_err) const;

  double get_kkt_solution(VectorXd& z, VectorXd& lambda) const;

  friend class DistanceProblem;

 private:
  // KKT ODE functions.
  void KktOde_(VectorXd& z_t, VectorXd& lambda_t);

  // Convex sets.
  const std::shared_ptr<ConvexSet> C1_;
  const std::shared_ptr<ConvexSet> C2_;
  // Solver options.
  const std::shared_ptr<SolverOptions> opt_;
  // Ipopt minimum distance problem.
  Ipopt::SmartPtr<Ipopt::TNLP> prob_;
  const std::shared_ptr<DistanceSolver> solver_;
  // KKT solution.
  VectorXd z_opt_;
  VectorXd lambda_opt_;
  double dist2_opt_;
  // Hessian of the cost function.
  MatrixXd hess_cost_;
  // Cholesky decomposition objects.
  Eigen::LDLT<MatrixXd> kkt_ldlt_;
};

}  // namespace sccbf

#endif  // SCCBF_COLLISION_COLLISION_PAIR_H_