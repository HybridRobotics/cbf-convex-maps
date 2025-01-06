#ifndef SCCBF_SYSTEM_QUADROTOR_REDUCED_H_
#define SCCBF_SYSTEM_QUADROTOR_REDUCED_H_

#include <Eigen/Core>

#include "sccbf/data_types.h"
#include "sccbf/system/dynamical_system.h"

namespace sccbf {

class QuadrotorReduced : public DynamicalSystem {
 public:
  QuadrotorReduced(double mass, const MatrixXd& constr_mat_u,
                   const VectorXd& constr_vec_u);

  ~QuadrotorReduced();

  void Dynamics(const VectorXd& x, VectorXd& f, MatrixXd& g) const override;

  int nx() const override;

  int nu() const override;

  int nru() const override;

  const VectorXd& IntegrateDynamics(const VectorXd& u, double dt) override;

 private:
  static constexpr double kGravity = 9.81;  // [m/s^2].

  static constexpr int kNx = 15;  // (x, v, R).
  static constexpr int kNu = 4;   // (F, w).

  const int nru_;

  const double mass_;
  const MatrixXd inertia_;
  MatrixXd inertia_inv_;
};

inline int QuadrotorReduced::nx() const { return kNx; }

inline int QuadrotorReduced::nu() const { return kNu; }

inline int QuadrotorReduced::nru() const { return nru_; }

}  // namespace sccbf

#endif  // SCCBF_SYSTEM_QUADROTOR_REDUCED_H_
