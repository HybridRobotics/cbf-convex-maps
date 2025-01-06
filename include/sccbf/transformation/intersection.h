#ifndef SCCBF_TRANSFORMATION_INTERSECTION_H_
#define SCCBF_TRANSFORMATION_INTERSECTION_H_

#include <Eigen/Core>
#include <cassert>
#include <memory>

#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

struct Derivatives;

class IntersectionSet : public ConvexSet {
 public:
  IntersectionSet(const std::shared_ptr<ConvexSet>& C1,
                  const std::shared_ptr<ConvexSet>& C2,
                  const MatrixXd& hess_lb);

  ~IntersectionSet();

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

  void set_x(const VectorXd& x) override;

  void set_dx(const VectorXd& dx) override;

 private:
  // Convex sets.
  const std::shared_ptr<ConvexSet> C1_;
  const std::shared_ptr<ConvexSet> C2_;
  // Hessian lower bound matrix.
  const MatrixXd hess_lb_;
};

inline int IntersectionSet::dim() const { return C1_->dim(); }

inline int IntersectionSet::nz() const { return C1_->nz(); }

inline int IntersectionSet::nr() const { return C1_->nr() + C2_->nr(); }

inline int IntersectionSet::nx() const { return C1_->nx(); }

inline int IntersectionSet::ndx() const { return C1_->ndx(); }

inline MatrixXd IntersectionSet::get_projection_matrix() const {
  return C1_->get_projection_matrix();
}

inline bool IntersectionSet::is_strongly_convex() const {
  return C1_->is_strongly_convex() & C2_->is_strongly_convex();
}

inline void IntersectionSet::set_x(const VectorXd& x) {
  assert(x.rows() == nx());
  x_ = x;
  C1_->set_x(x);
  C2_->set_x(x);
}

inline void IntersectionSet::set_dx(const VectorXd& dx) {
  assert(dx.rows() == ndx());
  dx_ = dx;
  C1_->set_dx(dx);
  C2_->set_dx(dx);
}

}  // namespace sccbf

#endif  // SCCBF_TRANSFORMATION_INTERSECTION_H_
