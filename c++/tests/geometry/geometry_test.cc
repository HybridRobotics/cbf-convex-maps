#include <gtest/gtest.h>

#include <Eigen/Core>
#include <memory>

#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/geometry/ellipsoid.h"
#include "sccbf/geometry/polytope.h"
#include "sccbf/geometry/quadrotor_corridor.h"
#include "sccbf/geometry/quadrotor_downwash.h"
#include "sccbf/geometry/quadrotor_shape.h"
#include "sccbf/geometry/quadrotor_uncertainty.h"
#include "sccbf/geometry/static_ellipsoid.h"
#include "sccbf/geometry/static_polytope.h"
#include "sccbf/system/quadrotor_reduced.h"
#include "sccbf/transformation/intersection.h"
#include "sccbf/transformation/minkowski.h"
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

StateVariables RandomQuadStateVariables(const ConvexSet& set) {
  constexpr int nx = 15;
  assert(set.nx() == nx);
  assert(set.ndx() == nx);
  const int nz = set.nz();
  const int nr = set.nr();
  StateVariables var(nx, nx, nz, nr);

  // Set x.
  var.x.head<6>() = VectorXd::Random(6);
  MatrixXd R(3, 3);
  RandomRotation<3>(R);
  var.x.tail<9>() = R.reshaped(9, 1);
  // Set dx and x_dot.
  var.dx.head<3>() = var.x.segment<3>(3);
  var.dx.segment<3>(3) = VectorXd::Random(3);
  VectorXd w = VectorXd::Random(3);
  MatrixXd w_hat = MatrixXd::Zero(3, 3);
  HatMap<3>(w, w_hat);
  var.dx.tail<9>() = (R * w_hat).reshaped(9, 1);
  var.x_dot = var.dx;

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

testing::AssertionResult AssertMatrixEQ(const char* mat1_expr,
                                        const char* mat2_expr,
                                        const char* /*tol_expr*/,
                                        const MatrixXd& mat1,
                                        const MatrixXd& mat2, double tol) {
  const double inf_norm = (mat1 - mat2).lpNorm<Eigen::Infinity>();
  if (inf_norm < tol) return testing::AssertionSuccess();

  return testing::AssertionFailure()
         << mat1_expr << " =" << std::endl
         << mat1.format(kMatFmt) << std::endl
         << "and " << mat2_expr << " = " << std::endl
         << mat2.format(kMatFmt) << std::endl
         << "are not equal";
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

// IntersectionSet test
TEST(ConvexSetTest, IntersectionSet) {
  // Set 1: polytope.
  const int nr = 10;
  VectorXd center = VectorXd::Zero(3);
  MatrixXd A(nr, 3);
  VectorXd b(nr);
  const double in_radius = 1.0;
  RandomPolytope(center, in_radius, A, b);
  const double sc_modulus = 1e-2;
  std::shared_ptr<ConvexSet> ptr1 =
      std::make_shared<Polytope3d>(A, b, 0.0, sc_modulus, true);
  // Set 2: ellipsoid.
  MatrixXd Q(3, 3);
  const double eps = 1.0;
  RandomSpdMatrix(Q, eps);
  std::shared_ptr<ConvexSet> ptr2 = std::make_shared<Ellipsoid3d>(Q, 0.0);
  // Intersection set.
  const MatrixXd hess_lb = sc_modulus * MatrixXd::Identity(3, 3);
  std::shared_ptr<ConvexSet> set_ptr =
      std::make_shared<IntersectionSet>(ptr1, ptr2, hess_lb);

  StateVariables var = RandomSeStateVariables<3>(*set_ptr);

  const Derivatives& d1 =
      set_ptr->UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(*set_ptr, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);
}

// MinkowskiSumSet test
TEST(ConvexSetTest, MinkowskiSumSet) {
  // Set 1: polytope.
  const int nr = 10;
  VectorXd center = VectorXd::Zero(3);
  MatrixXd A(nr, 3);
  VectorXd b(nr);
  const double in_radius = 1.0;
  RandomPolytope(center, in_radius, A, b);
  const double sc_modulus = 1e-2;
  std::shared_ptr<ConvexSet> ptr1 =
      std::make_shared<Polytope3d>(A, b, 0.0, sc_modulus, true);
  // Set 2: ellipsoid.
  MatrixXd Q(3, 3);
  const double eps = 1.0;
  RandomSpdMatrix(Q, eps);
  std::shared_ptr<ConvexSet> ptr2 = std::make_shared<Ellipsoid3d>(Q, 0.0);
  // Intersection set.
  std::shared_ptr<ConvexSet> set_ptr =
      std::make_shared<MinkowskiSumSet>(ptr1, ptr2);

  StateVariables var = RandomSeStateVariables<3>(*set_ptr);

  const Derivatives& d1 =
      set_ptr->UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(*set_ptr, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);
}

// QuadrotorShape test
TEST(GeometryTest, QuadrotorShape) {
  const double pow = 2.5;
  const Eigen::Vector4d coeff(1.0, 1.0, 0.4, 0.3);
  const double margin = 0;
  auto set = QuadrotorShape(pow, coeff, margin);

  StateVariables var = RandomQuadStateVariables(set);

  // Derivative test.
  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);

  // Lie derivative test.
  const int nu = 4;
  const double mass = 0.5;  // [kg].
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);

  std::shared_ptr<DynamicalSystem> sys =
      std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u);
  MatrixXd fg(15, nu + 1);
  sys->Dynamics(fg);

  MatrixXd L_fg_y1(1, nu + 1), L_fg_y2(1, nu + 1);
  set.LieDerivatives(var.x, var.z, var.y, fg, L_fg_y1);
  NumericalLieDerivatives(set, var.x, var.z, var.y, fg, fg, L_fg_y2);

  EXPECT_PRED_FORMAT3(AssertMatrixEQ, L_fg_y1, L_fg_y2, kDerivativeErrorTol);
}

// QuadrotorDownwash test
TEST(GeometryTest, QuadrotorDownwash) {
  MatrixXd A(5, 3);
  A << 4.0, 0.0, 2.0,  //
      0.0, 4.0, 2.0,   //
      -4.0, 0.0, 2.0,  //
      0.0, -4.0, 2.0,  //
      0.0, 0.0, -1.5;
  const VectorXd b = VectorXd::Zero(5);
  const double level = 1.5;
  const double margin = 0;
  auto set = QuadrotorDownwash(A, b, level, margin);

  StateVariables var = RandomQuadStateVariables(set);

  // Derivative test.
  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);

  // Lie derivative test.
  const int nu = 4;
  const double mass = 0.5;  // [kg].
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);

  std::shared_ptr<DynamicalSystem> sys =
      std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u);
  MatrixXd fg(15, nu + 1);
  sys->Dynamics(fg);

  MatrixXd L_fg_y1(1, nu + 1), L_fg_y2(1, nu + 1);
  set.LieDerivatives(var.x, var.z, var.y, fg, L_fg_y1);
  NumericalLieDerivatives(set, var.x, var.z, var.y, fg, fg, L_fg_y2);

  EXPECT_PRED_FORMAT3(AssertMatrixEQ, L_fg_y1, L_fg_y2, kDerivativeErrorTol);
}

// QuadrotorCorridor test
TEST(GeometryTest, QuadrotorCorridor) {
  const double stop_time = 2.0;         // [s].
  const double orientation_cost = 1.0;  // [s].
  const double max_vel = 1.0;           // [m/s].
  const double margin = 0;
  auto set = QuadrotorCorridor(stop_time, orientation_cost, max_vel, margin);

  StateVariables var = RandomQuadStateVariables(set);

  // Derivative test.
  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var,
                      10 * kDerivativeErrorTol);

  // Lie derivative test.
  const int nu = 4;
  const double mass = 0.5;  // [kg].
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);

  std::shared_ptr<DynamicalSystem> sys =
      std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u);
  MatrixXd fg(15, nu + 1);
  sys->Dynamics(fg);

  MatrixXd L_fg_y1(1, nu + 1), L_fg_y2(1, nu + 1);
  set.LieDerivatives(var.x, var.z, var.y, fg, L_fg_y1);
  NumericalLieDerivatives(set, var.x, var.z, var.y, fg, fg, L_fg_y2);

  EXPECT_PRED_FORMAT3(AssertMatrixEQ, L_fg_y1, L_fg_y2,
                      10 * kDerivativeErrorTol);
}

// QuadrotorUncertainty test
TEST(GeometryTest, QuadrotorUncertainty) {
  const MatrixXd mat = MatrixXd::Random(3, 3);
  const double eps = 1.0;
  MatrixXd Q = mat.transpose() * mat + eps * MatrixXd::Identity(3, 3);
  Eigen::Vector4d coeff = Eigen::Vector4d::Random();
  coeff(0) = coeff(0) + 1.1;
  const double margin = 0;
  auto set = QuadrotorUncertainty(Q, coeff, margin);

  StateVariables var = RandomQuadStateVariables(set);

  // Derivative test.
  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.x_dot, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);

  // Lie derivative test.
  const int nu = 4;
  const double mass = 0.5;  // [kg].
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);

  std::shared_ptr<DynamicalSystem> sys =
      std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u);
  MatrixXd fg(15, nu + 1);
  sys->Dynamics(fg);

  MatrixXd L_fg_y1(1, nu + 1), L_fg_y2(1, nu + 1);
  set.LieDerivatives(var.x, var.z, var.y, fg, L_fg_y1);
  NumericalLieDerivatives(set, var.x, var.z, var.y, fg, fg, L_fg_y2);

  EXPECT_PRED_FORMAT3(AssertMatrixEQ, L_fg_y1, L_fg_y2, kDerivativeErrorTol);
}

}  // namespace
