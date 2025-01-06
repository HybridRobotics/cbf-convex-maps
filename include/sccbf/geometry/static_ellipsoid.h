#ifndef SCCBF_GEOMETRY_STATIC_ELLIPSOID_H_
#define SCCBF_GEOMETRY_STATIC_ELLIPSOID_H_

#include <Eigen/Core>

#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

struct Derivatives;

template <int nz_>
class StaticEllipsoid : public ConvexSet {
 public:
  StaticEllipsoid(const MatrixXd& Q, const VectorXd& p, double margin);

  ~StaticEllipsoid();

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
  static constexpr int kNz = nz_;
  static constexpr int kNDim = kNz;
  static constexpr int kNx = 0;
  static constexpr int kNdx = 0;
  static constexpr int kNr = 1;

  const MatrixXd Q_;
  const VectorXd p_;
};

template <int nz_>
inline int StaticEllipsoid<nz_>::dim() const {
  return kNDim;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::nz() const {
  return kNz;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::nr() const {
  return kNr;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::nx() const {
  return kNx;
}

template <int nz_>
inline int StaticEllipsoid<nz_>::ndx() const {
  return kNdx;
}

template <int nz_>
inline MatrixXd StaticEllipsoid<nz_>::get_projection_matrix() const {
  return MatrixXd::Identity(kNz, kNz);
}

template <int nz_>
inline bool StaticEllipsoid<nz_>::is_strongly_convex() const {
  return true;
}

typedef StaticEllipsoid<2> StaticEllipsoid2d;
typedef StaticEllipsoid<3> StaticEllipsoid3d;
typedef StaticEllipsoid<4> StaticEllipsoid4d;

}  // namespace sccbf

#endif  // SCCBF_GEOMETRY_STATIC_ELLIPSOID_H_
