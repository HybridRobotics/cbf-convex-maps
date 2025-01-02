#include <gtest/gtest.h>

#include <Eigen/Core>
#include <memory>

#include "quadrotor_corridor.h"
#include "quadrotor_downwash.h"
#include "quadrotor_shape.h"
#include "quadrotor_uncertainty.h"
#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/system/quadrotor_reduced.h"
#include "sccbf/utils/matrix_utils.h"
#include "sccbf/utils/numerical_derivatives.h"

namespace {

using namespace sccbf;

// Quadrotor state variables struct, and random state variables
struct Variables {
  VectorXd x;
  VectorXd dx;
  VectorXd z;
  VectorXd y;

  Variables(int nx, int nz, int nr) : x(nx), dx(nx), z(nz), y(nr) {}
};

Variables RandomVariables(const ConvexSet& set) {
  constexpr int nx = 15;
  assert(set.nx() == nx);
  assert(set.ndx() == nx);
  const int nz = set.nz();
  const int nr = set.nr();
  Variables var(nx, nz, nr);

  // Set x.
  var.x.head<6>() = VectorXd::Random(6);
  MatrixXd R(3, 3);
  RandomRotation<3>(R);
  var.x.tail<9>() = R.reshaped(9, 1);
  // Set dx.
  var.dx.head<3>() = var.x.segment<3>(3);
  var.dx.segment<3>(3) = VectorXd::Random(3);
  VectorXd w = VectorXd::Random(3);
  MatrixXd w_hat = MatrixXd::Zero(3, 3);
  HatMap<3>(w, w_hat);
  var.dx.tail<9>() = (R * w_hat).reshaped(9, 1);

  // Set z, y.
  var.z = VectorXd::Random(nz);
  var.y = VectorXd::Random(nr);
  var.y.array() = var.y.array() + 1;

  return var;
}

// Assertion functions
Eigen::IOFormat kVecFmt(4, Eigen::DontAlignCols, ", ", "\n", "[", "]");
Eigen::IOFormat kMatFmt(4, 0, ", ", "\n", "[", "]");

testing::AssertionResult AssertDerivativeEQ(
    const char* d1_expr, const char* d2_expr, const char* /*var_expr*/,
    const char* /*tol_expr*/, const Derivatives& d1, const Derivatives& d2,
    const Variables& var, double tol) {
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

// QuadrotorShape test
TEST(GeometryTest, QuadrotorShape) {
  const double pow = 2.5;
  const Eigen::Vector4d coeff(1.0, 1.0, 0.4, 0.3);
  const double margin = 0;
  auto set = QuadrotorShape(pow, coeff, margin);

  Variables var = RandomVariables(set);

  // Derivative test.
  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.dx, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);

  // Lie derivative test.
  const int nu = 4;
  const double mass = 0.5;  // [kg].
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);

  std::shared_ptr<DynamicalSystem> sys =
      std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u);
  VectorXd f(15);
  MatrixXd g(15, nu);
  sys->Dynamics(var.x, f, g);
  MatrixXd fg(15, nu + 1);
  fg.col(0) = f;
  fg.rightCols(nu) = g;

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

  Variables var = RandomVariables(set);

  // Derivative test.
  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.dx, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);

  // Lie derivative test.
  const int nu = 4;
  const double mass = 0.5;  // [kg].
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);

  std::shared_ptr<DynamicalSystem> sys =
      std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u);
  VectorXd f(15);
  MatrixXd g(15, nu);
  sys->Dynamics(var.x, f, g);
  MatrixXd fg(15, nu + 1);
  fg.col(0) = f;
  fg.rightCols(nu) = g;

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

  Variables var = RandomVariables(set);

  // Derivative test.
  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.dx, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);

  // Lie derivative test.
  const int nu = 4;
  const double mass = 0.5;  // [kg].
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);

  std::shared_ptr<DynamicalSystem> sys =
      std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u);
  VectorXd f(15);
  MatrixXd g(15, nu);
  sys->Dynamics(var.x, f, g);
  MatrixXd fg(15, nu + 1);
  fg.col(0) = f;
  fg.rightCols(nu) = g;

  MatrixXd L_fg_y1(1, nu + 1), L_fg_y2(1, nu + 1);
  set.LieDerivatives(var.x, var.z, var.y, fg, L_fg_y1);
  NumericalLieDerivatives(set, var.x, var.z, var.y, fg, fg, L_fg_y2);

  EXPECT_PRED_FORMAT3(AssertMatrixEQ, L_fg_y1, L_fg_y2, kDerivativeErrorTol);
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

  Variables var = RandomVariables(set);

  // Derivative test.
  const Derivatives& d1 =
      set.UpdateDerivatives(var.x, var.dx, var.z, var.y, kFlag);
  Derivatives d2 =
      NumericalDerivatives(set, var.x, var.dx, var.dx, var.z, var.y);

  EXPECT_PRED_FORMAT4(AssertDerivativeEQ, d1, d2, var, kDerivativeErrorTol);

  // Lie derivative test.
  const int nu = 4;
  const double mass = 0.5;  // [kg].
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);

  std::shared_ptr<DynamicalSystem> sys =
      std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u);
  VectorXd f(15);
  MatrixXd g(15, nu);
  sys->Dynamics(var.x, f, g);
  MatrixXd fg(15, nu + 1);
  fg.col(0) = f;
  fg.rightCols(nu) = g;

  MatrixXd L_fg_y1(1, nu + 1), L_fg_y2(1, nu + 1);
  set.LieDerivatives(var.x, var.z, var.y, fg, L_fg_y1);
  NumericalLieDerivatives(set, var.x, var.z, var.y, fg, fg, L_fg_y2);

  EXPECT_PRED_FORMAT3(AssertMatrixEQ, L_fg_y1, L_fg_y2, kDerivativeErrorTol);
}

}  // namespace
