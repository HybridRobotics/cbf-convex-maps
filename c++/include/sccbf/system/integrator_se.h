#ifndef SCCBF_SYSTEM_INTEGRATOR_SE_H_
#define SCCBF_SYSTEM_INTEGRATOR_SE_H_

#include <Eigen/Core>

#include "sccbf/data_types.h"
#include "sccbf/system/dynamical_system.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

template <int n>
class IntegratorSe : public DynamicalSystem {
 public:
  IntegratorSe(const MatrixXd& constr_mat_u, const VectorXd& constr_vec_u);

  ~IntegratorSe();

  void Dynamics(const VectorXd& x, VectorXd& f, MatrixXd& g) const override;

  int nx() const override;

  int nu() const override;

  int nru() const override;

  const VectorXd& IntegrateDynamics(const VectorXd& u, double dt);

 private:
  static constexpr int kNx = n + n * n;
  static constexpr int kNu = (n == 2) ? 3 : 6;

  const int nru_;
};

template <int n>
IntegratorSe<n>::IntegratorSe(const MatrixXd& constr_mat_u,
                              const VectorXd& constr_vec_u)
    : DynamicalSystem(kNx, kNu, constr_mat_u, constr_vec_u),
      nru_(static_cast<int>(constr_mat_u.rows())) {
  static_assert((n == 2) || (n == 3));

  CheckDimensions();
}

template <int n>
IntegratorSe<n>::~IntegratorSe() {}

template <int n>
void IntegratorSe<n>::Dynamics(const VectorXd& x, VectorXd& f,
                               MatrixXd& g) const {
  assert(x.rows() == kNx);
  assert(f.rows() == kNx);
  assert((g.rows() == kNx) && (g.cols() == kNu));

  f = VectorXd::Zero(kNx);
  g = MatrixXd::Zero(kNx, kNu);
  g.topLeftCorner<n, n>() = MatrixXd::Identity(n, n);
  if constexpr (n == 2) {
    const double ctheta = x(2);
    const double stheta = x(3);
    g(2, 2) = -stheta;
    g(3, 2) = ctheta;
    g(4, 2) = -ctheta;
    g(5, 2) = -stheta;
  }
  if constexpr (n == 3) {
    const auto R = x.tail<9>().reshaped(3, 3);
    g.block<3, 1>(3, 4) = -R.col(2);
    g.block<3, 1>(3, 5) = R.col(1);
    g.block<3, 1>(6, 3) = R.col(2);
    g.block<3, 1>(6, 5) = -R.col(0);
    g.block<3, 1>(9, 3) = -R.col(1);
    g.block<3, 1>(9, 4) = R.col(0);
  }
}

template <int n>
inline int IntegratorSe<n>::nx() const {
  return kNx;
}

template <int n>
inline int IntegratorSe<n>::nu() const {
  return kNu;
}

template <int n>
inline int IntegratorSe<n>::nru() const {
  return nru_;
}

template <int n>
const VectorXd& IntegratorSe<n>::IntegrateDynamics(const VectorXd& u,
                                                   double dt) {
  const auto p = x_.head<n>();
  const auto R = x_.tail<n * n>().reshaped(3, 3);

  const auto v = u.head<n>();
  const auto w = u.tail<kNu - n>();
  x_.head<n>() = p + v * dt;
  MatrixXd R_new(n, n);
  IntegrateSo(R, w * dt, R_new);
  x_.tail<n * n>() = R_new.reshaped(n * n, 1);
  return x_;
}

typedef IntegratorSe<2> IntegratorSe2;
typedef IntegratorSe<3> IntegratorSe3;

}  // namespace sccbf

#endif  // SCCBF_SYSTEM_INTEGRATOR_SE_H_
