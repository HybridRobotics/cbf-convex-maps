#ifndef SCCBF_GEOMETRY_CONVEX_SET_H_
#define SCCBF_GEOMETRY_CONVEX_SET_H_

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"

namespace sccbf {

class ConvexSet {
 protected:
  ConvexSet(int nz, int nr, double margin_)
      : derivatives(nz, nr), margin(margin_) {}
  
  Derivatives derivatives;
  double margin;

 public:
  virtual ~ConvexSet() {}

  virtual const Derivatives& update_derivatives(const VectorXd& x,
                                                const VectorXd& dx,
                                                const VectorXd& z,
                                                const VectorXd& y,
                                                DFlags f) = 0;

  virtual void lie_derivatives_x(const VectorXd& x, const VectorXd& z,
                                 const VectorXd& y, const MatrixXd& fg,
                                 MatrixXd& L_fgA_y) const = 0;

  virtual int dim() const = 0;

  virtual int nz() const = 0;

  virtual int nr() const = 0;

  virtual int nx() const = 0;

  virtual int ndx() const = 0;

  // post_transformation_matrix (allows for projections, user must ensure
  // compactness when needed).
  virtual const MatrixXd& projection_matrix() const = 0;

  // Hessian sparsity pattern.
  virtual const MatrixXd& hessian_sparsity_pattern() const = 0;

  virtual bool is_strongly_convex() const = 0;

  inline double safety_margin() const { return margin; };
};

}  // namespace sccbf

#endif  // SCCBF_GEOMETRY_CONVEX_SET_H_