#include "sccbf/transformation/intersection.h"

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/utils/matrix_utils.h"

namespace sccbf {

IntersectionSet::IntersectionSet(const std::shared_ptr<ConvexSet>& C1,
                                 const std::shared_ptr<ConvexSet>& C2,
                                 const MatrixXd& hess_lb)
    : ConvexSet(C1->nz(), C1->nr() + C2->nr(), C1->nx(), C1->ndx(),
                std::max(C1->get_safety_margin(), C2->get_safety_margin())),
      C1_(C1),
      C2_(C2),
      hess_lb_(hess_lb) {
  assert(C1_->dim() == C2_->dim());
  assert(C1_->nz() == C2_->nz());
  assert(C1_->nx() == C2_->nx());
  assert(C1_->ndx() == C2_->ndx());
  if (C1_->is_strongly_convex() && C2_->is_strongly_convex()) {
    if (!IsPositiveDefinite(hess_lb)) {
      throw std::runtime_error("Hessian lower bound is not positive definite!");
    }
  }
  const auto P1 = C1_->get_projection_matrix();
  const auto P2 = C2_->get_projection_matrix();
  assert((P1 - P2).lpNorm<Eigen::Infinity>() < 1e-5);

  CheckDimensions();
}

IntersectionSet::~IntersectionSet() {}

const Derivatives& IntersectionSet::UpdateDerivatives(const VectorXd& x,
                                                      const VectorXd& dx,
                                                      const VectorXd& z,
                                                      const VectorXd& y,
                                                      DerivativeFlags flag) {
  assert(x.rows() == nx());
  assert(dx.rows() == ndx());
  assert(z.rows() == nz());
  assert(y.rows() == nr());

  const int nr1 = C1_->nr();
  const int nr2 = C2_->nr();
  const auto y1 = y.head(nr1);
  const auto y2 = y.tail(nr2);
  const Derivatives& d1 = C1_->UpdateDerivatives(x, dx, z, y1, flag);
  const Derivatives& d2 = C2_->UpdateDerivatives(x, dx, z, y2, flag);

  if (has_flag(flag, DerivativeFlags::f)) {
    derivatives_.f.head(nr1) = d1.f;
    derivatives_.f.tail(nr2) = d2.f;
  }
  if (has_flag(flag, DerivativeFlags::f_z)) {
    derivatives_.f_z.topRows(nr1) = d1.f_z;
    derivatives_.f_z.bottomRows(nr2) = d2.f_z;
  }
  if (has_flag(flag, DerivativeFlags::f_x)) {
    derivatives_.f_x.head(nr1) = d1.f_x;
    derivatives_.f_x.tail(nr2) = d2.f_x;
  }
  if (has_flag(flag, DerivativeFlags::f_zz_y) ||
      has_flag(flag, DerivativeFlags::f_zz_y_lb)) {
    derivatives_.f_zz_y = d1.f_zz_y + d2.f_zz_y;
    derivatives_.f_zz_y_lb = hess_lb_;
  }
  if (has_flag(flag, DerivativeFlags::f_xz_y)) {
    derivatives_.f_xz_y = d1.f_xz_y + d2.f_xz_y;
  }
  return derivatives_;
}

void IntersectionSet::LieDerivatives(const VectorXd& x, const VectorXd& z,
                                     const VectorXd& y, const MatrixXd& fg,
                                     MatrixXd& L_fg_y) const {
  assert(x.rows() == nx());
  assert(z.rows() == nz());
  assert(y.rows() == nr());
  assert(fg.rows() == ndx());
  assert(L_fg_y.rows() == 1);
  assert(L_fg_y.cols() == fg.cols());

  const int nr1 = C1_->nr();
  const int nr2 = C2_->nr();
  const auto y1 = y.head(nr1);
  const auto y2 = y.tail(nr2);

  C1_->LieDerivatives(x, z, y1, fg, L_fg_y);
  MatrixXd L_fg_y2 = MatrixXd::Zero(1, fg.cols());
  C2_->LieDerivatives(x, z, y2, fg, L_fg_y2);
  L_fg_y = L_fg_y + L_fg_y2;
}

}  // namespace sccbf
