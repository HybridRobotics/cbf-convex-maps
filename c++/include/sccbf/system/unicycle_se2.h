#ifndef SCCBF_SYSTEM_UNICYCLE_SE2_H_
#define SCCBF_SYSTEM_UNICYCLE_SE2_H_

#include <Eigen/Core>

#include "sccbf/data_types.h"
#include "sccbf/system/dynamical_system.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

class UnicycleSe2 : public DynamicalSystem {
 public:
  UnicycleSe2(const MatrixXd& constr_mat_u, const VectorXd& constr_vec_u);

  ~UnicycleSe2();

  void Dynamics(const VectorXd& x, VectorXd& f, MatrixXd& g) const override;

  int nx() const override;

  int nu() const override;

  int nru() const override;

  const VectorXd& IntegrateDynamics(const VectorXd& u, double dt) override;

 private:
  static constexpr int kNx = 2 + 4;
  static constexpr int kNu = 2;

  const int nru_;
};

inline UnicycleSe2::UnicycleSe2(const MatrixXd& constr_mat_u,
                                const VectorXd& constr_vec_u)
    : DynamicalSystem(kNx, kNu, constr_mat_u, constr_vec_u),
      nru_(static_cast<int>(constr_mat_u.rows())) {
  CheckDimensions();
}

inline UnicycleSe2::~UnicycleSe2() {}

inline void UnicycleSe2::Dynamics(const VectorXd& x, VectorXd& f,
                                  MatrixXd& g) const {
  assert(x.rows() == kNx);
  assert(f.rows() == kNx);
  assert((g.rows() == kNx) && (g.cols() == kNu));

  f = VectorXd::Zero(kNx);
  const double ctheta = x(2);
  const double stheta = x(3);
  g = MatrixXd::Zero(kNx, kNu);
  g(0, 0) = ctheta;
  g(1, 0) = stheta;
  g(2, 1) = -stheta;
  g(3, 1) = ctheta;
  g(4, 1) = -ctheta;
  g(5, 1) = -stheta;
}

inline int UnicycleSe2::nx() const { return kNx; }

inline int UnicycleSe2::nu() const { return kNu; }

inline int UnicycleSe2::nru() const { return nru_; }

inline const VectorXd& UnicycleSe2::IntegrateDynamics(const VectorXd& u,
                                                      double dt) {
  const auto p = x_.head<2>();
  const auto R = x_.tail<4>().reshaped(2, 2);

  const double v = u(0);
  const double w = u(1);
  x_.head<2>() = p + R.col(0) * v * dt;
  MatrixXd R_new(2, 2);
  IntegrateSo2(R, w * dt, R_new);
  x_.tail<4>() = R_new.reshaped(4, 1);
  return x_;
}

}  // namespace sccbf

#endif  // SCCBF_SYSTEM_UNICYCLE_SE2_H_
