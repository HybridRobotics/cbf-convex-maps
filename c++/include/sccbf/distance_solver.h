#ifndef SCCBF_DISTANCE_SOLVER_H_
#define SCCBF_DISTANCE_SOLVER_H_

#include <IpIpoptApplication.hpp>
#include <IpTNLP.hpp>
#include <memory>

#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

class DistanceProblem : public Ipopt::TNLP {
 public:
  DistanceProblem(const std::shared_ptr<ConvexSet>& C1, const VectorXd& x1,
                  const std::shared_ptr<ConvexSet>& C2, const VectorXd& x2,
                  const MatrixXd& M);

  ~DistanceProblem();

  void set_states(const VectorXd& x1, const VectorXd& x2);

  // Virtual functions.
  bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
                    Ipopt::Index& nnz_h_lag,
                    IndexStyleEnum& index_style) override;

  bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                       Ipopt::Index m, Ipopt::Number* g_l,
                       Ipopt::Number* g_u) override;

  bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                          bool /*init_z*/, Ipopt::Number* /*z_L*/,
                          Ipopt::Number* /*z_U*/, Ipopt::Index m,
                          bool init_lambda, Ipopt::Number* lambda) override;

  bool eval_f(Ipopt::Index /*n*/, const Ipopt::Number* x, bool /*new_x*/,
              Ipopt::Number& obj_value) override;

  bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                   Ipopt::Number* grad_f) override;

  bool eval_g(Ipopt::Index /*n*/, const Ipopt::Number* x, bool /*new_x*/,
              Ipopt::Index m, Ipopt::Number* g) override;

  bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                  Ipopt::Index m, Ipopt::Index /*nele_jac*/, Ipopt::Index* iRow,
                  Ipopt::Index* jCol, Ipopt::Number* values) override;

  bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
              Ipopt::Number obj_factor, Ipopt::Index /*m*/,
              const Ipopt::Number* lambda, bool /*new_lambda*/,
              Ipopt::Index /*nele_hess*/, Ipopt::Index* iRow,
              Ipopt::Index* jCol, Ipopt::Number* values) override;

  void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n,
                         const Ipopt::Number* x, const Ipopt::Number* /*z_L*/,
                         const Ipopt::Number* /*z_U*/, Ipopt::Index m,
                         const Ipopt::Number* /*g*/,
                         const Ipopt::Number* lambda, Ipopt::Number obj_value,
                         const Ipopt::IpoptData* /*ip_data*/,
                         Ipopt::IpoptCalculatedQuantities* /*ip_cq*/
                         ) override;

  DistanceProblem(const DistanceProblem&) = delete;

  DistanceProblem& operator=(const DistanceProblem&) = delete;

 private:
  const std::shared_ptr<ConvexSet> C1_;
  const std::shared_ptr<ConvexSet> C2_;
  VectorXd x1_;
  VectorXd x2_;
  MatrixXd M_;
  VectorXd z_;
  VectorXd lambda_;
  double dist2_;
};

class DistanceSolver {
 public:
  DistanceSolver();

  double MinimumDistance(DistanceProblem& prob);

  ~DistanceSolver();

  void SetOption(const std::string& name, const std::string& value);

  void SetOption(const std::string& name, int value);

  void SetOption(const std::string& name, double value);

  double GetTotalWallclockTime();

  DistanceSolver(const DistanceSolver&) = delete;

  DistanceSolver& operator=(const DistanceSolver&) = delete;

 private:
  std::unique_ptr<Ipopt::IpoptApplication> app_;
};

}  // namespace sccbf

#endif  // SCCBF_DISTANCE_SOLVER_H_
