#ifndef SCCBF_TRANSFORMATION_MINKOWSKI_H_
#define SCCBF_TRANSFORMATION_MINKOWSKI_H_

#include <Eigen/Core>
#include <cassert>
#include <memory>

#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

struct Derivatives;

class MinkowskiSumSet : public ConvexSet {
 public:
  MinkowskiSumSet(const std::shared_ptr<ConvexSet>& C1,
                  const std::shared_ptr<ConvexSet>& C2);

  ~MinkowskiSumSet();

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
};

inline int MinkowskiSumSet::dim() const { return C1_->dim(); }

inline int MinkowskiSumSet::nz() const { return C1_->nz() + C2_->nz(); }

inline int MinkowskiSumSet::nr() const { return C1_->nr() + C2_->nr(); }

inline int MinkowskiSumSet::nx() const { return C1_->nx(); }

inline int MinkowskiSumSet::ndx() const { return C1_->ndx(); }

inline MatrixXd MinkowskiSumSet::get_projection_matrix() const {
  const int nz1 = C1_->nz();
  const int nz2 = C2_->nz();

  MatrixXd P = MatrixXd::Zero(dim(), nz());
  P.leftCols(nz1) = C1_->get_projection_matrix();
  P.rightCols(nz2) = C2_->get_projection_matrix();
  return P;
}

inline bool MinkowskiSumSet::is_strongly_convex() const {
  return C1_->is_strongly_convex() & C2_->is_strongly_convex();
}

inline void MinkowskiSumSet::set_x(const VectorXd& x) {
  assert(x.rows() == nx());
  x_ = x;
  C1_->set_x(x);
  C2_->set_x(x);
}

inline void MinkowskiSumSet::set_dx(const VectorXd& dx) {
  assert(dx.rows() == ndx());
  dx_ = dx;
  C1_->set_dx(dx);
  C2_->set_dx(dx);
}

}  // namespace sccbf

#endif  // SCCBF_TRANSFORMATION_MINKOWSKI_H_
