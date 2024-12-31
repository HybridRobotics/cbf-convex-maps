#ifndef APPS_QUADROTOR_SHAPE_H_
#define APPS_QUADROTOR_SHAPE_H_

#include <Eigen/Core>

#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

struct Derivatives;

class QuadrotorShape : public ConvexSet {
 public:
  QuadrotorShape(double pow, const Eigen::Vector4d& coeff, double margin);

  ~QuadrotorShape();

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

  const double pow_;
  const Eigen::Vector4d coeff_;
  Eigen::Vector4d coeff_pow_;
};

inline int QuadrotorShape::dim() const { return kNDim; }

inline int QuadrotorShape::nz() const { return kNz; }

inline int QuadrotorShape::nr() const { return kNr; }

inline int QuadrotorShape::nx() const { return kNx; }

inline int QuadrotorShape::ndx() const { return kNdx; }

inline MatrixXd QuadrotorShape::get_projection_matrix() const {
  return MatrixXd::Identity(kNz, kNz);
}

inline bool QuadrotorShape::is_strongly_convex() const { return true; }

}  // namespace sccbf

#endif  // APPS_QUADROTOR_SHAPE_H_
