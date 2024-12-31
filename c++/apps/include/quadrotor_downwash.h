#ifndef APPS_QUADROTOR_DOWNWASH_H_
#define APPS_QUADROTOR_DOWNWASH_H_

#include <Eigen/Core>

#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

struct Derivatives;

class QuadrotorDownwash : public ConvexSet {
 public:
  QuadrotorDownwash(const MatrixXd& A, const VectorXd& b, double level,
                    double margin);

  ~QuadrotorDownwash();

  const Derivatives& UpdateDerivatives(const VectorXd& x, const VectorXd& dx,
                                       const VectorXd& z, const VectorXd& y,
                                       DerivativeFlags flag) override;

  void LieDerivatives(const VectorXd& x, const VectorXd& z, const VectorXd& y,
                      const MatrixXd& fg, MatrixXd& L_fg_y) const override;

  int dim() const override;

  int nz() const override;

  int nr() const override;

  int nx() const override;

  int ndx() const override;

  MatrixXd get_projection_matrix() const override;

  bool is_strongly_convex() const override;

 private:
  static constexpr int kNz = 3;
  static constexpr int kNDim = 3;
  static constexpr int kNx = 3 + 3 + 9;
  static constexpr int kNdx = kNx;
  static constexpr int kNr = 1;

  const MatrixXd A_;
  const VectorXd b_;
  const double level_;
};

inline int QuadrotorDownwash::dim() const { return kNDim; }

inline int QuadrotorDownwash::nz() const { return kNz; }

inline int QuadrotorDownwash::nr() const { return kNr; }

inline int QuadrotorDownwash::nx() const { return kNx; }

inline int QuadrotorDownwash::ndx() const { return kNdx; }

inline MatrixXd QuadrotorDownwash::get_projection_matrix() const {
  return MatrixXd::Identity(kNz, kNz);
}

inline bool QuadrotorDownwash::is_strongly_convex() const { return true; }

}  // namespace sccbf

#endif  // APPS_QUADROTOR_DOWNWASH_H_
