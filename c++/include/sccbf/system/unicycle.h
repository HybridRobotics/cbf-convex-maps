#ifndef SCCBF_SYSTEM_UNICYCLE_H_
#define SCCBF_SYSTEM_UNICYCLE_H_

#include <Eigen/Core>
#include <cmath>

#include "sccbf/data_types.h"
#include "sccbf/system/dynamical_system.h"

namespace sccbf {

class Unicycle : public DynamicalSystem {
 public:
  Unicycle(const MatrixXd& constr_mat_u, const VectorXd& constr_vec_u);

  ~Unicycle();

  void Dynamics(const VectorXd& x, VectorXd& f, MatrixXd& g) const override;

  int nx() const override;

  int nu() const override;

  int nru() const override;

 private:
  static constexpr int kNx = 3;
  static constexpr int kNu = 2;

  const int nru_;
};

inline Unicycle::Unicycle(const MatrixXd& constr_mat_u,
                          const VectorXd& constr_vec_u)
    : DynamicalSystem(kNx, kNu, constr_mat_u, constr_vec_u),
      nru_(static_cast<int>(constr_mat_u.rows())) {
  CheckDimensions();
}

inline Unicycle::~Unicycle() {}

inline void Unicycle::Dynamics(const VectorXd& x, VectorXd& f,
                               MatrixXd& g) const {
  assert(x.rows() == kNx);
  assert(f.rows() == kNx);
  assert((g.rows() == kNx) && (g.cols() == kNu));

  f = VectorXd::Zero(kNx);
  MatrixXd g(kNx, kNu);
  const double theta = x(2);
  g << std::cos(theta), 0.0, std::sin(theta), 0.0, 0.0, 1.0;
}

inline int Unicycle::nx() const { return kNx; }

inline int Unicycle::nu() const { return kNu; }

inline int Unicycle::nru() const { return nru_; }

}  // namespace sccbf

#endif  // SCCBF_SYSTEM_UNICYCLE_H_
