#ifndef SCCBF_SYSTEM_DYNAMICAL_SYSTEM_H_
#define SCCBF_SYSTEM_DYNAMICAL_SYSTEM_H_

#include <cassert>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"

namespace sccbf {

class DynamicalSystem {
 protected:
  DynamicalSystem(int nx, int nu, int nru);

  DynamicalSystem(int nx, int nu, const MatrixXd& constr_mat_u,
                  const VectorXd& constr_vec_u);

  VectorXd x_;
  MatrixXd constr_mat_u_;
  VectorXd constr_vec_u_;

 public:
  // Virtual functions.
  virtual ~DynamicalSystem() {}

  virtual void Dynamics(const VectorXd& x, VectorXd& f, MatrixXd& g) const = 0;

  virtual int nx() const = 0;

  virtual int nu() const = 0;

  virtual int nru() const = 0;

  // Non-virtual functions.
  void Dynamics(VectorXd& f, MatrixXd& g) const;

  const VectorXd& IntegrateDynamics(const VectorXd& u, double dt);

  void set_x(const VectorXd& x);

  const VectorXd& x() const;

  void set_input_constraints(const MatrixXd& constr_mat_u,
                             const VectorXd& constr_vec_u);

  const MatrixXd& constr_mat_u() const;

  const VectorXd& constr_vec_u() const;

  void CheckDimensions() const;
};

inline DynamicalSystem::DynamicalSystem(int nx, int nu, int nru)
    : x_(nx), constr_mat_u_(nru, nu), constr_vec_u_(nru) {
  x_ = VectorXd::Zero(nx);
  constr_mat_u_ = MatrixXd::Zero(nru, nu);
  constr_vec_u_ = VectorXd::Zero(nru);
}

inline DynamicalSystem::DynamicalSystem(int nx, int nu,
                                        const MatrixXd& constr_mat_u,
                                        const VectorXd& constr_vec_u)
    : x_(nx) {
  assert(constr_mat_u.cols() == nu);
  assert(constr_mat_u.rows() == constr_vec_u.rows());
  x_ = VectorXd::Zero(nx);
  constr_mat_u_ = constr_mat_u;
  constr_vec_u_ = constr_vec_u;
}

inline void DynamicalSystem::Dynamics(VectorXd& f, MatrixXd& g) const {
  return Dynamics(x_, f, g);
}

inline const VectorXd& DynamicalSystem::IntegrateDynamics(const VectorXd& u,
                                                          double dt) {
  assert(u.rows() == nu());
  // Allow negative times (for integration purposes).
  // assert(dt >= 0);

  // Allow input infeasibility.
  // const auto input_infeasibility = constr_mat_u_ * u - constr_vec_u_;
  // if (!(input_infeasibility.array() <= 0.0).all())
  //   std::runtime_error("Input is not feasible for the input constraints!");

  VectorXd f = VectorXd::Zero(nx());
  MatrixXd g = MatrixXd::Zero(nx(), nu());
  Dynamics(f, g);
  const auto dx = f + g * u;
  return x_ = x_ + dx * dt;
}

inline void DynamicalSystem::set_x(const VectorXd& x) {
  assert(x.rows() == nx());
  x_ = x;
}

inline const VectorXd& DynamicalSystem::x() const { return x_; }

inline void DynamicalSystem::set_input_constraints(
    const MatrixXd& constr_mat_u, const VectorXd& constr_vec_u) {
  assert((constr_mat_u.rows() == nru()) && (constr_mat_u.cols() == nu()));
  assert(constr_vec_u.rows() == nru());
  constr_mat_u_ = constr_mat_u;
  constr_vec_u_ = constr_vec_u;
}

inline const MatrixXd& DynamicalSystem::constr_mat_u() const {
  return constr_mat_u_;
}

inline const VectorXd& DynamicalSystem::constr_vec_u() const {
  return constr_vec_u_;
}

inline void DynamicalSystem::CheckDimensions() const {
  assert(nx() >= 0);
  assert(nu() >= 0);
  assert(nru() >= 0);

  assert(x_.rows() == nx());

  assert((constr_mat_u_.rows() == nru()) && (constr_mat_u_.cols() == nu()));
  assert(constr_vec_u_.rows() == nru());
}

}  // namespace sccbf

#endif  // SCCBF_SYSTEM_DYNAMICAL_SYSTEM_H_