#ifndef SCCBF_GEOMETRY_CONVEX_SET_H_
#define SCCBF_GEOMETRY_CONVEX_SET_H_

#include <cassert>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"

namespace sccbf {

class ConvexSet {
 protected:
  ConvexSet(int nz, int nr, int nx, int ndx, double margin);

  VectorXd x_;
  VectorXd dx_;
  Derivatives derivatives_;
  double margin_;

 public:
  // Virtual functions.
  virtual ~ConvexSet() {}

  virtual const Derivatives& UpdateDerivatives(const VectorXd& x,
                                               const VectorXd& dx,
                                               const VectorXd& z,
                                               const VectorXd& y,
                                               DerivativeFlags flag) = 0;

  virtual void LieDerivatives(const VectorXd& x, const VectorXd& z,
                              const VectorXd& y, const MatrixXd& fg,
                              MatrixXd& L_fg_y) const = 0;

  virtual int dim() const = 0;

  virtual int nz() const = 0;

  virtual int nr() const = 0;

  virtual int nx() const = 0;

  virtual int ndx() const = 0;

  // post_transformation_matrix (allows for projections, user must ensure
  // compactness when needed).
  virtual MatrixXd get_projection_matrix() const = 0;

  virtual bool is_strongly_convex() const = 0;

  // Non-virtual functions.
  const Derivatives& UpdateDerivatives(const VectorXd& z, const VectorXd& y,
                                       DerivativeFlags flag);

  const Derivatives& get_derivatives() const;

  void set_x(const VectorXd& x);

  void set_dx(const VectorXd& dx);

  void set_states(const VectorXd& x, const VectorXd& dx);

  const VectorXd& x();

  const VectorXd& dx();

  double get_safety_margin() const;

  void CheckDimensions() const;
};

inline ConvexSet::ConvexSet(int nz, int nr, int nx, int ndx, double margin)
    : x_(nx), dx_(ndx), derivatives_(nz, nr), margin_(margin) {
  x_ = VectorXd::Zero(nx);
  dx_ = VectorXd::Zero(ndx);
}

inline const Derivatives& ConvexSet::UpdateDerivatives(const VectorXd& z,
                                                       const VectorXd& y,
                                                       DerivativeFlags flag) {
  return UpdateDerivatives(x_, dx_, z, y, flag);
}

inline const Derivatives& ConvexSet::get_derivatives() const {
  return derivatives_;
}

inline void ConvexSet::set_x(const VectorXd& x) {
  assert(x.rows() == nx());
  x_ = x;
}

inline void ConvexSet::set_dx(const VectorXd& dx) {
  assert(dx.rows() == ndx());
  dx_ = dx;
}

inline void ConvexSet::set_states(const VectorXd& x, const VectorXd& dx) {
  set_x(x);
  set_dx(dx);
}

inline const VectorXd& ConvexSet::x() { return x_; }

inline const VectorXd& ConvexSet::dx() { return dx_; }

inline double ConvexSet::get_safety_margin() const { return margin_; }

inline void ConvexSet::CheckDimensions() const {
  assert(dim() >= 0);
  // ConvexSet instance must be solid (i.e. have a nonempty interior).
  assert(dim() <= nz());
  assert(nr() >= 0);
  assert(nx() >= 0);
  assert(ndx() >= 0);

  assert(x_.rows() == nx());
  assert(dx_.rows() == ndx());

  const MatrixXd mat = get_projection_matrix();
  assert(mat.rows() == dim());
  assert(mat.cols() == nz());

  assert(margin_ >= 0);
}

}  // namespace sccbf

#endif  // SCCBF_GEOMETRY_CONVEX_SET_H_