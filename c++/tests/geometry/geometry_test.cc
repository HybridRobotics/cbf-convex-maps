#include <gtest/gtest.h>

#include <Eigen/Core>
#include <memory>

#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/geometry/ellipsoid.h"
#include "sccbf/geometry/polytope.h"
#include "sccbf/geometry/static_ellipsoid.h"
#include "sccbf/geometry/static_polytope.h"
#include "sccbf/utils/matrix_utils.h"
#include "sccbf/utils/numerical_derivatives.h"

namespace {

using namespace sccbf;

// State variables struct, and random state variables
struct StateVariables {
  VectorXd x;
  VectorXd x_dot;
  VectorXd dx;
  VectorXd z;
  VectorXd y;

  StateVariables(int nx, int ndx, int nz, int nr)
      : x(nx), x_dot(nx), dx(ndx), z(nz), y(nr) {}
};

StateVariables RandomStaticStateVariables(const ConvexSet& set) {
  constexpr int nx = 0;
  constexpr int ndx = 0;
  assert(set.nx() == nx);
  assert(set.ndx() == ndx);
  const int nz = set.nz();
  const int nr = set.nr();
  StateVariables var(nx, ndx, nz, nr);

  // Set x.
  var.x = VectorXd::Zero(nx);
  var.x_dot = VectorXd::Zero(nx);
  var.dx = VectorXd::Zero(ndx);
  // Set z, y.
  var.z = VectorXd::Random(nz);
  var.y = VectorXd::Random(nr);
  var.y.array() = var.y.array() + 1;

  return var;
}

template <int nse>
StateVariables RandomSeStateVariables(const ConvexSet& set) {
  static_assert((nse == 2) || (nse == 3));
  constexpr int nx = nse + nse * nse;
  constexpr int ndx = (nse == 2) ? 3 : 6;
  assert(set.nx() == nx);
  assert(set.ndx() == ndx);
  const int nz = set.nz();
  const int nr = set.nr();
  StateVariables var(nx, ndx, nz, nr);

  // Set x.
  var.x.head<nse>() = VectorXd::Random(nse);
  MatrixXd R(nse, nse);
  RandomRotation<nse>(R);
  var.x.tail<nse * nse>() = R.reshaped(nse * nse, 1);
  // Set v.
  var.dx.head<nse>() = var.x_dot.head<nse>() = VectorXd::Random(nse);
  // Set w.
  if constexpr (nse == 2) {
    var.dx.tail<1>() = VectorXd::Random(1);
    MatrixXd dR(2, 2);
    dR.col(0) = R.col(1) * var.dx(2);
    dR.col(1) = -R.col(0) * var.dx(2);
    var.x_dot.tail<4>() = dR.reshaped(4, 1);
  }
  if constexpr (nse == 3) {
    var.dx.tail<3>() = VectorXd::Random(3);
    MatrixXd w_hat = MatrixXd::Zero(3, 3);
    HatMap<3>(var.dx.tail<3>(), w_hat);
    var.x_dot.tail<9>() = (R * w_hat).reshaped(9, 1);
  }

  // Set z, y.
  var.z = VectorXd::Random(nz);
  var.y = VectorXd::Random(nr);
  var.y.array() = var.y.array() + 1;

  return var;
}

// Assertion function
Eigen::IOFormat kVecFmt(4, Eigen::DontAlignCols, ", ", "\n", "[", "]");
Eigen::IOFormat kMatFmt(4, 0, ", ", "\n", "[", "]");

testing::AssertionResult AssertDerivativeEQ(
    const char* d1_expr, const char* d2_expr, const char* /*var_expr*/,
    const char* /*tol_expr*/, const Derivatives& d1, const Derivatives& d2,
    const StateVariables& var, double tol) {
  auto failure = testing::AssertionFailure();

  auto print_term = [&d1_expr, &d2_expr, &failure](const char* term_expr,
                                                   const MatrixXd& term1,
                                                   const MatrixXd& term2) {
    failure << d1_expr << "." << term_expr << " = " << std::endl
            << term1.format(kMatFmt) << std::endl
            << "is not equal to " << d2_expr << "." << term_expr << " = "
            << std::endl
            << term2.format(kMatFmt) << std::endl;
  };

  bool success = true;

  if ((d1.f - d2.f).lpNorm<Eigen::Infinity>() >= tol) {
    success = false;
    print_term("f", d1.f, d2.f);
  }
  if ((d1.f_x - d2.f_x).lpNorm<Eigen::Infinity>() >= tol) {
    success = false;
    print_term("f_x", d1.f_x, d2.f_x);
  }
  if ((d1.f_z - d2.f_z).lpNorm<Eigen::Infinity>() >= tol) {
    success = false;
    print_term("f_z", d1.f_z, d2.f_z);
  }
  if ((d1.f_xz_y - d2.f_xz_y).lpNorm<Eigen::Infinity>() >= tol) {
    success = false;
    print_term("f_xz_y", d1.f_xz_y, d2.f_xz_y);
  }
  if ((d1.f_zz_y - d2.f_zz_y).lpNorm<Eigen::Infinity>() >= tol) {
    success = false;
    print_term("f_zz_y", d1.f_zz_y, d2.f_zz_y);
  }

  if (success) return testing::AssertionSuccess();

  failure << "Failure at x = " << std::endl
          << var.x.transpose().format(kVecFmt) << std::endl
          << ", dx = " << std::endl
          << var.dx.transpose().format(kVecFmt) << std::endl
          << ", z = " << std::endl
          << var.z.transpose().format(kVecFmt) << std::endl
          << ", y = " << std::endl
          << var.y.transpose().format(kVecFmt);

  return failure;
}

// Geometry tests
const DerivativeFlags kFlag = DerivativeFlags::f | DerivativeFlags::f_x |
                              DerivativeFlags::f_z | DerivativeFlags::f_xz_y |
                              DerivativeFlags::f_zz_y;
const double kDerivativeErrorTol = 1e-3;

// Ellipsoid test
TEST(ConvexSetTest, Ellipsoid2d) {
  MatrixXd Q(2, 2);
  const double eps = 1.0;
  RandomSpdMatrix(Q, eps);
  auto set = Ellipsoid2d(Q, 0.0);
  StateVariables var = RandomSeStateVariables<2>(set);

  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);
}

TEST(ConvexSetTest, Ellipsoid3d) {
  MatrixXd Q(3, 3);
  const double eps = 1.0;
  RandomSpdMatrix(Q, eps);
  auto set = Ellipsoid3d(Q, 0.0);
  StateVariables var = RandomSeStateVariables<3>(set);

  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);
}

// StaticEllipsoid test
TEST(ConvexSetTest, StaticEllipsoid2d) {
  MatrixXd Q(2, 2);
  const double eps = 1.0;
  RandomSpdMatrix(Q, eps);
  VectorXd p = VectorXd::Random(2);
  auto set = StaticEllipsoid2d(Q, p, 0.0);
  StateVariables var = RandomStaticStateVariables(set);

  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);
}

TEST(ConvexSetTest, StaticEllipsoid3d) {
  MatrixXd Q(3, 3);
  const double eps = 1.0;
  RandomSpdMatrix(Q, eps);
  VectorXd p = VectorXd::Random(3);
  auto set = StaticEllipsoid3d(Q, p, 0.0);
  StateVariables var = RandomStaticStateVariables(set);

  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);
}

// Polytope test
TEST(ConvexSetTest, Polytope2d) {
  const int nr = 6;
  VectorXd center = VectorXd::Zero(2);
  MatrixXd A(nr, 2);
  VectorXd b(nr);
  const double in_radius = 1;
  RandomPolytope(center, in_radius, A, b);
  const double sc_modulus = 1e-2;
  auto set = Polytope2d(A, b, 0.0, sc_modulus, true);
  StateVariables var = RandomSeStateVariables<2>(set);

  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);
}

TEST(ConvexSetTest, Polytope3d) {
  const int nr = 10;
  VectorXd center = VectorXd::Zero(3);
  MatrixXd A(nr, 3);
  VectorXd b(nr);
  const double in_radius = 3;
  RandomPolytope(center, in_radius, A, b);
  const double sc_modulus = 1e-2;
  auto set = Polytope3d(A, b, 0.0, sc_modulus, true);
  StateVariables var = RandomSeStateVariables<3>(set);

  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);
}

// StaticPolytope test
TEST(ConvexSetTest, StaticPolytope2d) {
  const int nr = 6;
  VectorXd p = VectorXd::Random(2);
  MatrixXd A(nr, 2);
  VectorXd b(nr);
  const double in_radius = 1;
  RandomPolytope(p, in_radius, A, b);
  const double sc_modulus = 1e-2;
  auto set = StaticPolytope2d(A, b, p, 0.0, sc_modulus, true);
  StateVariables var = RandomStaticStateVariables(set);

  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);
}

TEST(ConvexSetTest, StaticHalfspace) {
  const int nr = 1;
  VectorXd p = VectorXd::Random(3);
  MatrixXd A(nr, 3);
  VectorXd b(nr);
  const double in_radius = 3;
  RandomPolytope(p, in_radius, A, b);
  const double sc_modulus = 0;
  auto set = StaticPolytope3d(A, b, p, 0.0, sc_modulus, true);
  StateVariables var = RandomStaticStateVariables(set);

  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);
}

TEST(ConvexSetTest, StaticPolytope3d) {
  const int nr = 10;
  VectorXd p = VectorXd::Random(3);
  MatrixXd A(nr, 3);
  VectorXd b(nr);
  const double in_radius = 3;
  RandomPolytope(p, in_radius, A, b);
  const double sc_modulus = 1e-2;
  auto set = StaticPolytope3d(A, b, p, 0.0, sc_modulus, true);
  StateVariables var = RandomStaticStateVariables(set);

  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);
}

}  // namespace
