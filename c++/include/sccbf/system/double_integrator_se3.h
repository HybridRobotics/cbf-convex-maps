#ifndef SCCBF_SYSTEM_DOUBLE_INTEGRATOR_SE3_H_
#define SCCBF_SYSTEM_DOUBLE_INTEGRATOR_SE3_H_

#include <Eigen/Core>

#include "sccbf/data_types.h"
#include "sccbf/system/dynamical_system.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

class DoubleIntegratorSe3 : public DynamicalSystem {
 public:
  DoubleIntegratorSe3(const MatrixXd& constr_mat_u,
                      const VectorXd& constr_vec_u);

  ~DoubleIntegratorSe3();

  void Dynamics(const VectorXd& x, VectorXd& f, MatrixXd& g) const override;

  int nx() const override;

  int nu() const override;

  int nru() const override;

  const VectorXd& IntegrateDynamics(const VectorXd& u, double dt);

 private:
  static constexpr int kNx = (3 + 3) + 9;
  static constexpr int kNu = 3 + 3;

  const int nru_;
};

inline DoubleIntegratorSe3::DoubleIntegratorSe3(const MatrixXd& constr_mat_u,
                                                const VectorXd& constr_vec_u)
    : DynamicalSystem(kNx, kNu, constr_mat_u, constr_vec_u),
      nru_(static_cast<int>(constr_mat_u.rows())) {
  CheckDimensions();
}

inline DoubleIntegratorSe3::~DoubleIntegratorSe3() {}

inline void DoubleIntegratorSe3::Dynamics(const VectorXd& x, VectorXd& f,
                                          MatrixXd& g) const {
  assert(x.rows() == kNx);
  assert(f.rows() == kNx);
  assert((g.rows() == kNx) && (g.cols() == kNu));

  f = VectorXd::Zero(kNx);
  g = MatrixXd::Zero(kNx, kNu);
  g.block<3, 3>(3, 0) = MatrixXd::Identity(3, 3);
  const auto R = x.tail<9>().reshaped(3, 3);
  g.block<3, 1>(6, 4) = -R.col(2);
  g.block<3, 1>(6, 5) = R.col(1);
  g.block<3, 1>(9, 3) = R.col(2);
  g.block<3, 1>(9, 5) = -R.col(0);
  g.block<3, 1>(12, 3) = -R.col(1);
  g.block<3, 1>(12, 4) = R.col(0);
}

inline int DoubleIntegratorSe3::nx() const { return kNx; }

inline int DoubleIntegratorSe3::nu() const { return kNu; }

inline int DoubleIntegratorSe3::nru() const { return nru_; }

inline const VectorXd& DoubleIntegratorSe3::IntegrateDynamics(const VectorXd& u,
                                                              double dt) {
  const auto p = x_.head<3>();
  const auto R = x_.tail<9>().reshaped(3, 3);

  const auto v = u.head<3>();
  const auto w = u.tail<3>();
  x_.head<3>() = p + v * dt;
  MatrixXd R_new(3, 3);
  IntegrateSo3(R, w * dt, R_new);
  x_.tail<9>() = R_new.reshaped(9, 1);
  return x_;
}

}  // namespace sccbf

#endif  // SCCBF_SYSTEM_DOUBLE_INTEGRATOR_SE3_H_
