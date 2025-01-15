#ifndef APPS_QUADROTOR_CORRIDOR_H_
#define APPS_QUADROTOR_CORRIDOR_H_

#include <Eigen/Core>

#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

struct Derivatives;

class QuadrotorCorridor : public ConvexSet {
 public:
  QuadrotorCorridor(double stop_time, double orientation_const, double max_vel,
                    double margin);

  ~QuadrotorCorridor();

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

  static constexpr double kEps = 0.2;
  static constexpr double kEcc = 5;
  static constexpr double kMin = 0.1;  // [m]
  static constexpr double kQScale = 0.1;

  const double stop_time_;
  const double orientation_const_;
  const double max_vel_;
};

inline int QuadrotorCorridor::dim() const { return kNDim; }

inline int QuadrotorCorridor::nz() const { return kNz; }

inline int QuadrotorCorridor::nr() const { return kNr; }

inline int QuadrotorCorridor::nx() const { return kNx; }

inline int QuadrotorCorridor::ndx() const { return kNdx; }

inline MatrixXd QuadrotorCorridor::get_projection_matrix() const {
  return MatrixXd::Identity(kNz, kNz);
}

inline bool QuadrotorCorridor::is_strongly_convex() const { return true; }

}  // namespace sccbf

#endif  // APPS_QUADROTOR_CORRIDOR_H_
