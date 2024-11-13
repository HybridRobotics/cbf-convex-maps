#ifndef SCCBF_GEOMETRY_STATIC_POLYTOPE_H_
#define SCCBF_GEOMETRY_STATIC_POLYTOPE_H_

#include <Eigen/Core>

#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

struct Derivatives;

template <int nz_>
class StaticPolytope : public ConvexSet {
 public:
  StaticPolytope(const MatrixXd& A, const VectorXd& b, const VectorXd& p,
                 double margin, double sc_modulus, bool normalize);

  ~StaticPolytope();

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

  MatrixXd A_;
  VectorXd b_;
  const VectorXd& p_;
  const double sc_modulus_;
  const int nr_;
  bool strongly_convex_;
};

template <int nz_>
inline int StaticPolytope<nz_>::dim() const {
  return kNDim;
}

template <int nz_>
inline int StaticPolytope<nz_>::nz() const {
  return kNz;
}

template <int nz_>
inline int StaticPolytope<nz_>::nr() const {
  return nr_;
}

template <int nz_>
inline int StaticPolytope<nz_>::nx() const {
  return kNx;
}

template <int nz_>
inline int StaticPolytope<nz_>::ndx() const {
  return kNdx;
}

template <int nz_>
inline MatrixXd StaticPolytope<nz_>::get_projection_matrix() const {
  return MatrixXd::Identity(kNz, kNz);
}

template <int nz_>
inline bool StaticPolytope<nz_>::is_strongly_convex() const {
  return strongly_convex_;
}

typedef StaticPolytope<2> StaticPolytope2d;
typedef StaticPolytope<3> StaticPolytope3d;
typedef StaticPolytope<4> StaticPolytope4d;

}  // namespace sccbf

#endif  // SCCBF_GEOMETRY_STATIC_POLYTOPE_H_
