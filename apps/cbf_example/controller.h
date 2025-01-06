#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <vector>

#include "OsqpEigen/OsqpEigen.h"

// Data types
#include "sccbf/data_types.h"
// Geometries
#include "sccbf/geometry/convex_set.h"
// Systems
#include "sccbf/system/dynamical_system.h"
// Solver
#include "sccbf/collision/collision_pair.h"
#include "sccbf/solver_options.h"
// Utils
#include "sccbf/utils/control_utils.h"
#include "sccbf/utils/matrix_utils.h"

using namespace sccbf;

namespace {

const double kPi = static_cast<double>(EIGEN_PI);

struct Environment {
  // Collision pairs
  int num_obs;
  std::vector<std::shared_ptr<CollisionPair>> cps;
  // Quadrotor (and reduced-order) system
  std::shared_ptr<DynamicalSystem> sys;
  std::shared_ptr<DynamicalSystem> sys_r;
  double mass;
  MatrixXd inertia;
  double max_vel;
  // Solver options
  std::shared_ptr<SolverOptions> opt;

  void set_state(const VectorXd& x, const VectorXd& u);

  void set_state(const Eigen::Vector3d& p, const Eigen::Vector3d& v,
                 const Eigen::Matrix3d& R, const Eigen::Vector3d& w);

  void UpdateSystemState(const VectorXd& u, double dt);
};

inline void Environment::set_state(const VectorXd& x, const VectorXd& u) {
  assert(x.rows() == 15);
  assert(u.rows() == 4);

  sys_r->set_x(x);
  VectorXd f(15);
  MatrixXd g(15, 4);
  sys_r->Dynamics(f, g);
  const auto dx = f + g * u;

  for (auto cp : cps) cp->get_set2()->set_states(x, dx);
}

inline void Environment::set_state(const Eigen::Vector3d& p,
                                   const Eigen::Vector3d& v,
                                   const Eigen::Matrix3d& R,
                                   const Eigen::Vector3d& w) {
  VectorXd x(18);

  x.head<3>() = p;
  x.segment<3>(3) = v;
  x.segment<9>(6) = R.reshaped(9, 1);
  x.tail<3>() = w;
  sys->set_x(x);

  VectorXd u = VectorXd::Zero(4);
  u(0) = mass * 9.81;
  u.tail<3>() = w;
  const VectorXd xr = x.head<15>();
  set_state(xr, u);
}

inline void Environment::UpdateSystemState(const VectorXd& ur, double dt) {
  assert(ur.rows() == 4);

  // Convert reduced-order input to full system input
  const auto wd = ur.tail<3>();
  auto x = sys->x();
  const auto R = x.segment<9>(6).reshaped(3, 3);
  const VectorXd w = x.tail<3>();
  VectorXd u(4);
  u(0) = ur(0);
  VectorXd M(3);

  //  Controller 1: w-tracking control
  // double k_Omega = 1.0;
  // M = w.cross3(inertia * w) - k_Omega * inertia * (w - wd);

  //  Controller 2: R-tracking control
  MatrixXd Rd(3, 3);
  IntegrateSo3(R, wd * dt, Rd);

  const double scale = 1 / 0.0820 * inertia(0, 0);
  const double k_R = 8.81 * scale;
  const double k_Omega = 2.54 * scale;
  const So3PdParameters param = {k_R, k_Omega};
  // M = So3PdTrackingControl(R, Rd, w, wd, VectorXd::Zero(3), inertia, param);
  M = So3PdTrackingControl(R, Rd, w, VectorXd::Zero(3), VectorXd::Zero(3),
                           inertia, param);

  // Integrate full system
  u.tail<3>() = M;
  sys->IntegrateDynamics(u, dt);

  // Extract reduced-order state
  x = sys->x();
  const auto xr = x.head<15>();
  sys_r->set_x(xr);
}

class CbfQpController {
 public:
  CbfQpController(const Environment& env);

  ~CbfQpController() {};

  void Control(const Environment& env, const VectorXd& u_ref, VectorXd& u);

 private:
  static constexpr double kEps = 0.2;
  static constexpr double kVel = 0.1;          // < 1.0
  static constexpr double kMaxQuadAng = 60.0;  // [deg]

  static constexpr double kOmega = 1.0;
  static constexpr double kPrev = 0.0;
  static constexpr double kT = 1.0;
  static constexpr double kRef = 2.0;

  static constexpr double kAlphaVelCbf = 1.0;
  static constexpr double kAlphaAngCbf = 1.0;
  static constexpr double kAlphaDistCbf = 0.5;

  VectorXd margin2_;
  double mass_;
  double max_vel_;
  static constexpr double kMaxOmega = 3.14;  // [rad/s]
  int num_cons_;

  Eigen::SparseMatrix<double> hessian_;
  VectorXd gradient_;
  Eigen::SparseMatrix<double> constraint_mat_;
  VectorXd constraint_ub_;
  VectorXd constraint_lb_;
  OsqpEigen::Solver solver_;
};

inline CbfQpController::CbfQpController(const Environment& env) {
  // Set constants
  const double kGravity = 9.81;  // [m/s^2]
  mass_ = env.mass;
  max_vel_ = env.max_vel;
  const int num_cps = 2 * env.num_obs;
  margin2_.resize(num_cps);
  for (int i = 0; i < num_cps; ++i)
    margin2_(i) = std::pow(env.cps[i]->get_margin(), 2);
  num_cons_ = (num_cps + 1 + 1) + (1 + 3);

  // Resize and set Hessian matrix
  hessian_.resize(4, 4);
  hessian_.insert(0, 0) = kT + kRef;
  for (int i = 1; i < 4; ++i) hessian_.insert(i, i) = kOmega + kPrev + kRef;

  // Resize gradient, constraint ub and lb, and constraint matrix
  gradient_.resize(4);
  constraint_lb_.resize(num_cons_);
  constraint_ub_.resize(num_cons_);
  constraint_mat_.resize(num_cons_, 4);

  // Set fixed constraints
  //  CBF constraint matrix and ub values
  for (int row = 0; row < num_cps + 2; ++row) {
    for (int col = 0; col < 4; ++col) constraint_mat_.insert(row, col) = 1.0;
    constraint_ub_(row) = std::numeric_limits<double>::infinity();
  }
  constraint_lb_.head(num_cps + 2) = VectorXd::Zero(num_cps + 2);
  //  Input constraints
  //    Thrust constraint
  constraint_mat_.insert(num_cps + 2, 0) = 1.0;
  constraint_lb_(num_cps + 2) = mass_ * kGravity / 10.0;
  constraint_ub_(num_cps + 2) = 2.0 * mass_ * kGravity;
  //    Angular velocity constraint
  for (int i = 0; i < 3; ++i) {
    constraint_mat_.insert(num_cps + 3 + i, 1 + i) = 1.0;
    constraint_lb_(num_cps + 3 + i) = -kMaxOmega;
    constraint_ub_(num_cps + 3 + i) = kMaxOmega;
  }

  // instantiate the solver
  solver_.settings()->setVerbosity(false);
  solver_.settings()->setWarmStart(true);

  solver_.data()->setNumberOfVariables(4);
  solver_.data()->setNumberOfConstraints(num_cons_);

  solver_.data()->setHessianMatrix(hessian_);
  solver_.data()->setGradient(gradient_);
  solver_.data()->setLinearConstraintsMatrix(constraint_mat_);
  solver_.data()->setLowerBound(constraint_lb_);
  solver_.data()->setUpperBound(constraint_ub_);

  solver_.initSolver();

  // Solve initial QP
  solver_.solveProblem();
}

inline void CbfQpController::Control(const Environment& env,
                                     const VectorXd& u_ref, VectorXd& u) {
  assert((u_ref.rows() == 4) && (u.rows() == 4));

  VectorXd kGravityVec(3);
  kGravityVec << 0.0, 0.0, 9.81;
  const int num_cps = 2 * env.num_obs;
  VectorXd e3 = VectorXd::Zero(3);
  e3(2) = 1.0;
  MatrixXd e3_hat = MatrixXd::Zero(3, 3);
  HatMap<3>(e3, e3_hat);

  const auto x = env.sys_r->x();
  const auto v = x.segment<3>(3);
  const auto R = x.segment<9>(6).reshaped(3, 3);
  const auto w = env.sys->x().tail<3>();

  // Cost gradient
  gradient_(0) = -(kT * mass_ * kGravityVec(2) + kRef * u_ref(0));
  gradient_.tail<3>() = -(kPrev * w + kRef * u_ref.tail<3>());

  // Strongly convex map CBF constraints
  const MatrixXd fg1(0, 1);
  MatrixXd fg2(15, 5);
  env.sys_r->Dynamics(fg2);
  MatrixXd L_fg_y1(1, 1), L_fg_y2(1, 5);
  for (int i = 0; i < num_cps; ++i) {
    L_fg_y2 = MatrixXd::Zero(1, 5);
    env.cps[i]->LieDerivatives(fg1, fg2, L_fg_y1, L_fg_y2);

    const double h_dist = env.cps[i]->get_minimum_distance() - margin2_(i);

    const double dist_cbf_lb = -kAlphaDistCbf * h_dist - L_fg_y2(0, 0);

    for (int j = 0; j < 4; ++j)
      constraint_mat_.coeffRef(i, j) = L_fg_y2(0, j + 1);
    constraint_lb_(i) = dist_cbf_lb;
  }

  // Velocity bound CBF constraint
  const double v_norm = std::sqrt(std::pow(kEps * max_vel_, 2) + v.dot(v));
  const double h_vel = (1 - kVel) * max_vel_ - kVel * v.dot(R.col(2)) - v_norm;

  const auto y_vel = kVel * R.col(2) + v / v_norm;
  VectorXd vel_cbf_mat = VectorXd::Zero(4);
  vel_cbf_mat(0) = -y_vel.dot(R.col(2)) / mass_;
  vel_cbf_mat.tail<3>() = -kVel * e3_hat * R.transpose() * v;
  const double vel_cbf_lb = -kAlphaVelCbf * h_vel - y_vel.dot(kGravityVec);

  for (int i = 0; i < 4; ++i)
    constraint_mat_.coeffRef(num_cps, i) = vel_cbf_mat(i);
  constraint_lb_(num_cps) = vel_cbf_lb;

  // Quadrotor angle bound CBF constraint
  const double cos_max_ang = std::cos(kMaxQuadAng / 180.0 * kPi);
  const double h_ang = R(2, 2) - cos_max_ang;

  VectorXd ang_cbf_mat = VectorXd::Zero(4);
  ang_cbf_mat.tail<3>() = e3_hat * R.transpose() * e3;
  const double ang_cbf_lb = -kAlphaAngCbf * h_ang;

  for (int i = 0; i < 4; ++i)
    constraint_mat_.coeffRef(num_cps + 1, i) = ang_cbf_mat(i);
  constraint_lb_(num_cps + 1) = ang_cbf_lb;

  // Update QP
  solver_.updateGradient(gradient_);
  solver_.updateLinearConstraintsMatrix(constraint_mat_);
  solver_.updateLowerBound(constraint_lb_);

  // Solve QP;
  // If not solved to optimality, use backup control
  if (solver_.solveProblem() == OsqpEigen::ErrorExitFlag::NoError) {
    const VectorXd solution = solver_.getSolution();
    u = solution;
  } else {
    u(0) = u_ref(0);
    u.tail<3>() = So3PTrackingControl(R, MatrixXd::Identity(3, 3),
                                      VectorXd::Zero(3), {3.0, 0.0});
  }
}

}  // namespace
