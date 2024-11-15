#ifndef SCCBF_SYSTEM_INTEGRATOR_H_
#define SCCBF_SYSTEM_INTEGRATOR_H_

#include <Eigen/Core>

#include "sccbf/data_types.h"
#include "sccbf/system/dynamical_system.h"

namespace sccbf {

template <int n>
class Integrator : public DynamicalSystem {
 public:
  Integrator(const MatrixXd& constr_mat_u, const VectorXd& constr_vec_u);

  ~Integrator();

  void Dynamics(const VectorXd& x, VectorXd& f, MatrixXd& g) const override;

  int nx() const override;

  int nu() const override;

  int nru() const override;

 private:
  static constexpr int kNx = n;
  static constexpr int kNu = n;

  const int nru_;
};

template <int n>
Integrator<n>::Integrator(const MatrixXd& constr_mat_u,
                          const VectorXd& constr_vec_u)
    : DynamicalSystem(kNx, kNu, constr_mat_u, constr_vec_u),
      nru_(static_cast<int>(constr_mat_u.rows())) {
  static_assert(n >= 2);
  assert(constr_mat_u.cols() == kNu);

  CheckDimensions();
}

template <int n>
Integrator<n>::~Integrator() {}

template <int n>
inline void Integrator<n>::Dynamics(const VectorXd& x, VectorXd& f, MatrixXd& g) const {
  assert(x.rows() == kNx);
  assert(f.rows() == kNx);
  assert((g.rows() == kNx) && (g.cols() == kNu));

  f = VectorXd::Zero(kNx);
  g = MatrixXd::Identity(kNx, kNu);
}

template <int n>
inline int Integrator<n>::nx() const {return kNx;}

template <int n>
inline int Integrator<n>::nu() const {return kNu;}

template <int n>
inline int Integrator<n>::nru() const {return nru_;}

typedef Integrator<2> Integrator2d;
typedef Integrator<3> Integrator3d;
typedef Integrator<4> Integrator4d;

}  // namespace sccbf

#endif // SCCBF_SYSTEM_INTEGRATOR_H_
