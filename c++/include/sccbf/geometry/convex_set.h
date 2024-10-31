#ifndef SCCBF_GEOMETRY_CONVEX_SET_H_
#define SCCBF_GEOMETRY_CONVEX_SET_H_

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"

namespace sccbf {

class ConvexSet {
 protected:
  ConvexSet(int nz, int nr, double margin);

  Derivatives derivatives_;
  double margin_;

 public:
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
  virtual const MatrixXd& get_projection_matrix() const = 0;

  // Hessian sparsity pattern.
  virtual const MatrixXd& get_hessian_sparsity_matrix() const = 0;

  virtual bool is_strongly_convex() const = 0;

  double get_safety_margin() const;

  void CheckDimensions() const;

  const Derivatives& get_derivatives() const;
};

inline ConvexSet::ConvexSet(int nz, int nr, double margin)
    : derivatives_(nz, nr), margin_(margin) {}

inline double ConvexSet::get_safety_margin() const { return margin_; }

inline void ConvexSet::CheckDimensions() const {
  assert(dim() >= 0);
  // ConvexSet instance must be solid (i.e. have a nonempty interior).
  assert(dim() <= nz());
  assert(nr() >= 0);
  assert(nx() >= 0);
  assert(ndx() >= 0);

  const auto mat = get_projection_matrix();
  assert(mat.rows() == dim());
  assert(mat.cols() == nz());

  const auto hess = get_hessian_sparsity_matrix();
  assert(hess.rows() == nz());
  assert(hess.cols() == nz());

  assert(margin_ >= 0);
}

inline const Derivatives& ConvexSet::get_derivatives() const {
  return derivatives_;
}

}  // namespace sccbf

#endif  // SCCBF_GEOMETRY_CONVEX_SET_H_