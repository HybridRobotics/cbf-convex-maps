#ifndef SCCBF_SYSTEM_QUADROTOR_H_
#define SCCBF_SYSTEM_QUADROTOR_H_

#include <Eigen/Core>

#include "sccbf/data_types.h"
#include "sccbf/system/dynamical_system.h"

namespace sccbf {

class Quadrotor : public DynamicalSystem {
 public:
  Quadrotor(double mass, const MatrixXd& inertia, const MatrixXd& constr_mat_u,
            const VectorXd& constr_vec_u);

  ~Quadrotor();

  void Dynamics(const VectorXd& x, VectorXd& f, MatrixXd& g) const override;

  int nx() const override;

  int nu() const override;

  int nru() const override;

  const VectorXd& IntegrateDynamics(const VectorXd& u, double dt) override;

 private:
  static constexpr double kGravity = 9.81;  // [m/s^2].

  static constexpr int kNx = 18;  // (x, v, R, w).
  static constexpr int kNu = 4;   // (F, M).

  const int nru_;

  const double mass_;
  const MatrixXd inertia_;
  MatrixXd inertia_inv_;
};

inline int Quadrotor::nx() const { return kNx; }

inline int Quadrotor::nu() const { return kNu; }

inline int Quadrotor::nru() const { return nru_; }

}  // namespace sccbf

#endif  // SCCBF_SYSTEM_QUADROTOR_H_
