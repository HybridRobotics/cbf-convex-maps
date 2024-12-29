#include "sccbf/transformation/minkowski.h"

#include <Eigen/Core>
#include <cassert>

#include "sccbf/data_types.h"
#include "sccbf/derivatives.h"
#include "sccbf/geometry/convex_set.h"

namespace sccbf {

MinkowskiSumSet::MinkowskiSumSet(const std::shared_ptr<ConvexSet>& C1,
                                 const std::shared_ptr<ConvexSet>& C2)
    : ConvexSet(C1->nz() + C2->nz(), C1->nr() + C2->nr(), C1->nx(), C1->ndx(),
                C1->get_safety_margin() + C2->get_safety_margin()),
      C1_(C1),
      C2_(C2) {
  assert(C1_->dim() == C2_->dim());
  assert(C1_->nx() == C2_->nx());
  assert(C1_->ndx() == C2_->ndx());

  CheckDimensions();
}

MinkowskiSumSet::~MinkowskiSumSet() {}

const Derivatives& MinkowskiSumSet::UpdateDerivatives(const VectorXd& x,
                                                      const VectorXd& dx,
                                                      const VectorXd& z,
                                                      const VectorXd& y,
                                                      DerivativeFlags flag) {
  assert(x.rows() == nx());
  assert(dx.rows() == ndx());
  assert(z.rows() == nz());
  assert(y.rows() == nr());

  const int nz1 = C1_->nz();
  const int nz2 = C2_->nz();
  const int nr1 = C1_->nr();
  const int nr2 = C2_->nr();
  const auto z1 = z.head(nz1);
  const auto z2 = z.tail(nz2);
  const auto y1 = y.head(nr1);
  const auto y2 = y.tail(nr2);
  const Derivatives& d1 = C1_->UpdateDerivatives(x, dx, z1, y1, flag);
  const Derivatives& d2 = C2_->UpdateDerivatives(x, dx, z2, y2, flag);

  if (has_flag(flag, DerivativeFlags::f)) {
    derivatives_.f.head(nr1) = d1.f;
    derivatives_.f.tail(nr2) = d2.f;
  }
  if (has_flag(flag, DerivativeFlags::f_z)) {
    derivatives_.f_z.topLeftCorner(nr1, nz1) = d1.f_z;
    derivatives_.f_z.bottomRightCorner(nr2, nz2) = d2.f_z;
  }
  if (has_flag(flag, DerivativeFlags::f_x)) {
    derivatives_.f_x.head(nr1) = d1.f_x;
    derivatives_.f_x.tail(nr2) = d2.f_x;
  }
  if (has_flag(flag, DerivativeFlags::f_zz_y) ||
      has_flag(flag, DerivativeFlags::f_zz_y_lb)) {
    // Set f_zz_y.
    derivatives_.f_zz_y.topLeftCorner(nz1, nz1) = d1.f_zz_y;
    derivatives_.f_zz_y.bottomRightCorner(nz2, nz2) = d2.f_zz_y;
    derivatives_.f_zz_y.topRightCorner(nz1, nz2) = MatrixXd::Zero(nz1, nz2);
    derivatives_.f_zz_y.bottomLeftCorner(nz2, nz1) = MatrixXd::Zero(nz2, nz1);
    // Set f_zz_y_lb.
    derivatives_.f_zz_y_lb.topLeftCorner(nz1, nz1) = d1.f_zz_y_lb;
    derivatives_.f_zz_y_lb.bottomRightCorner(nz2, nz2) = d2.f_zz_y_lb;
    derivatives_.f_zz_y_lb.topRightCorner(nz1, nz2) = MatrixXd::Zero(nz1, nz2);
    derivatives_.f_zz_y_lb.bottomLeftCorner(nz2, nz1) =
        MatrixXd::Zero(nz2, nz1);
  }
  if (has_flag(flag, DerivativeFlags::f_xz_y)) {
    derivatives_.f_xz_y.head(nz1) = d1.f_xz_y;
    derivatives_.f_xz_y.tail(nz2) = d2.f_xz_y;
  }
  return derivatives_;
}

void MinkowskiSumSet::LieDerivatives(const VectorXd& x, const VectorXd& z,
                                     const VectorXd& y, const MatrixXd& fg,
                                     MatrixXd& L_fg_y) const {
  assert(x.rows() == nx());
  assert(z.rows() == nz());
  assert(y.rows() == nr());
  assert(fg.rows() == ndx());
  assert(L_fg_y.rows() == 1);
  assert(L_fg_y.cols() == fg.cols());

  const int nz1 = C1_->nz();
  const int nz2 = C2_->nz();
  const int nr1 = C1_->nr();
  const int nr2 = C2_->nr();
  const auto z1 = z.head(nz1);
  const auto z2 = z.tail(nz2);
  const auto y1 = y.head(nr1);
  const auto y2 = y.tail(nr2);

  C1_->LieDerivatives(x, z1, y1, fg, L_fg_y);
  MatrixXd L_fg_y2 = MatrixXd::Zero(1, fg.cols());
  C2_->LieDerivatives(x, z2, y2, fg, L_fg_y2);
  L_fg_y = L_fg_y + L_fg_y2;
}

}  // namespace sccbf
