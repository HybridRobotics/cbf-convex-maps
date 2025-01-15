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
const double kInf = std::numeric_limits<double>::infinity();
const double kGravity = 9.81;  // [m/s^2]

struct Environment {
  // Quadrotor-obstacle collision pairs
  int num_obs;
  MatrixXd centers;
  std::vector<std::shared_ptr<CollisionPair>> obs_cps;
  // (Reduced-order) quadrotor systems and inter-robot collision pairs
  int num_sys;
  std::vector<std::shared_ptr<DynamicalSystem>> sys_vec;
  std::vector<std::shared_ptr<CollisionPair>> sys_cps;
  double mass;
  double max_vel;
  // Solver options
  std::shared_ptr<SolverOptions> opt;

  void set_inputs(const VectorXd& u);

  void set_state(int i, const Eigen::Vector3d& p, const Eigen::Vector3d& v,
                 const Eigen::Matrix3d& R);

  void UpdateSystemState(const VectorXd& u, double dt);
};

inline void Environment::set_inputs(const VectorXd& u) {
  assert(u.rows() == num_sys * 4);

  VectorXd f(15);
  MatrixXd g(15, 4);
  if (num_obs > 0) {
    for (int i = 0; i < num_sys; ++i) {
      const auto xi = sys_vec[i]->x();
      const auto ui = u.segment<4>(4 * i);
      sys_vec[i]->Dynamics(f, g);
      const auto dxi = f + g * ui;
      obs_cps[num_obs * i]->get_set1()->set_states(xi, dxi);
    }
  }
  if (num_sys > 1) {
    for (int i = 0; i < num_sys; ++i) {
      const auto xi = sys_vec[i]->x();
      const auto ui = u.segment<4>(4 * i);
      sys_vec[i]->Dynamics(f, g);
      const auto dxi = f + g * ui;
      if (i == 0)
        sys_cps[0]->get_set1()->set_states(xi, dxi);
      else
        sys_cps[i - 1]->get_set2()->set_states(xi, dxi);
    }
  }
}

inline void Environment::set_state(int i, const Eigen::Vector3d& p,
                                   const Eigen::Vector3d& v,
                                   const Eigen::Matrix3d& R) {
  VectorXd x(15);
  x.head<3>() = p;
  x.segment<3>(3) = v;
  x.segment<9>(6) = R.reshaped(9, 1);
  sys_vec[i]->set_x(x);
}

inline void Environment::UpdateSystemState(const VectorXd& u, double dt) {
  assert(u.rows() == num_sys * 4);

  // Convert reduced-order input to full system input
  for (int i = 0; i < num_sys; ++i) {
    const auto ui = u.segment<4>(4 * i);
    sys_vec[i]->IntegrateDynamics(ui, dt);
  }
}

class CbfQpController {
 public:
  CbfQpController(const Environment& env);

  ~CbfQpController() {};

  void Control(const Environment& env, const VectorXd& u_ref, VectorXd& u);

  void get_margin2(VectorXd& margin2);

 private:
  static constexpr double kEps = 0.2;
  static constexpr double kVel = 0.1;          // < 1.0
  static constexpr double kMaxQuadAng = 60.0;  // [deg]
  static constexpr double kMaxOmega = 3.14;    // [rad/s]

  static constexpr double kOmega = 0.0;
  static constexpr double kT = 0.0;
  static constexpr double kRefT = 1.0;
  static constexpr double kRefOmega = 0.1;

  static constexpr double kAlphaVelCbf = 1.0;
  static constexpr double kAlphaAngCbf = 1.0;
  static constexpr double kAlphaDistCbf = 0.5;

  VectorXd margin2_;
  double mass_;
  double max_vel_;
  double cos_max_ang_;
  int num_sys_;
  int num_obs_;
  int num_obs_cps_;
  int num_sys_cps_;
  int num_cps_;
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
  mass_ = env.mass;
  max_vel_ = env.max_vel;
  cos_max_ang_ = std::cos(kMaxQuadAng / 180.0 * kPi);
  num_sys_ = env.num_sys;
  num_obs_ = env.num_obs;
  num_obs_cps_ = static_cast<int>(env.obs_cps.size());
  num_sys_cps_ = static_cast<int>(env.sys_cps.size());
  num_cps_ = num_obs_cps_ + num_sys_cps_;
  margin2_.resize(num_cps_);
  for (int i = 0; i < num_obs_cps_; ++i)
    margin2_(i) = std::pow(env.obs_cps[i]->get_margin(), 2);
  for (int i = 0; i < num_sys_cps_; ++i)
    margin2_(i + num_obs_cps_) = std::pow(env.sys_cps[i]->get_margin(), 2);
  num_cons_ = (num_cps_ + 2 * num_sys_) + (4 * num_sys_);

  // Resize and set Hessian matrix
  hessian_.resize(4 * num_sys_, 4 * num_sys_);
  for (int i = 0; i < num_sys_; ++i) {
    hessian_.insert(4 * i, 4 * i) = kT + kRefT;
    for (int j = 1; j < 4; ++j)
      hessian_.insert(4 * i + j, 4 * i + j) = kOmega + kRefOmega;
  }

  // Resize gradient, constraint ub and lb, and constraint matrix
  gradient_.resize(4 * num_sys_);
  gradient_ = VectorXd::Zero(4 * num_sys_);
  constraint_lb_.resize(num_cons_);
  constraint_ub_.resize(num_cons_);
  constraint_mat_.resize(num_cons_, 4 * num_sys_);

  // Set fixed constraints
  //  CBF constraint matrix and lb, ub values
  //    Quadrotor-obstacle collision pairs
  int idx = 0;
  for (int i = 0; i < num_sys_; ++i)
    for (int j = 0; j < num_obs_; ++j, ++idx)
      for (int k = 0; k < 4; ++k) constraint_mat_.insert(idx, 4 * i + k) = 1.0;
  //    Inter-quadrotor collision pairs
  for (int i = 0; i < num_sys_; ++i) {
    for (int j = i + 1; j < num_sys_; ++j, ++idx) {
      for (int k = 0; k < 4; ++k) {
        constraint_mat_.insert(idx, 4 * i + k) = 1.0;
        constraint_mat_.insert(idx, 4 * j + k) = 1.0;
      }
    }
  }
  //    Velocity, quadrotor-angle bounds
  for (int i = 0; i < num_sys_; ++i) {
    for (int j = 0; j < 4; ++j) {
      constraint_mat_.insert(i + num_cps_, 4 * i + j) = 1.0;
      constraint_mat_.insert(i + num_cps_ + num_sys_, 4 * i + j) = 1.0;
    }
  }
  constraint_ub_.head(num_cps_ + 2 * num_sys_) =
      kInf * VectorXd::Ones(num_cps_ + 2 * num_sys_);
  constraint_lb_.head(num_cps_ + 2 * num_sys_) =
      -kInf * VectorXd::Ones(num_cps_ + 2 * num_sys_);
  //  Input constraints (fixed)
  //    Thrust, angular velocity constraints
  idx = num_cps_ + 2 * num_sys_;
  for (int i = 0; i < num_sys_; ++i) {
    constraint_mat_.insert(idx + 4 * i, 4 * i) = 1.0;
    constraint_lb_(idx + 4 * i) = 0.0;
    constraint_ub_(idx + 4 * i) = 2.0 * mass_ * kGravity;
    for (int j = 1; j < 4; ++j) {
      constraint_mat_.insert(idx + 4 * i + j, 4 * i + j) = 1.0;
      constraint_lb_(idx + 4 * i + j) = -kMaxOmega;
      constraint_ub_(idx + 4 * i + j) = kMaxOmega;
    }
  }

  // instantiate the solver
  solver_.settings()->setVerbosity(false);
  solver_.settings()->setWarmStart(true);

  solver_.data()->setNumberOfVariables(4 * num_sys_);
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
  assert((u_ref.rows() == 4 * num_sys_) && (u.rows() == 4 * num_sys_));

  VectorXd kGravityVec(3);
  kGravityVec << 0.0, 0.0, kGravity;
  VectorXd e3 = VectorXd::Zero(3);
  e3(2) = 1.0;
  MatrixXd e3_hat = MatrixXd::Zero(3, 3);
  HatMap<3>(e3, e3_hat);

  // Cost gradient
  for (int i = 0; i < num_sys_; ++i) {
    const auto u_refi = u_ref.segment<4>(4 * i);
    gradient_(4 * i) = -(kT * mass_ * kGravity + kRefT * u_refi(0));
    gradient_.segment<3>(4 * i + 1) = -kRefOmega * u_refi.tail<3>();
  }

  bool use_backup_control = false;
  // Strongly convex map CBF constraints
  const MatrixXd fg_obs(0, 1);
  std::vector<MatrixXd> fg_sys(num_sys_);
  for (int i = 0; i < num_sys_; ++i) {
    MatrixXd fg(15, 5);
    env.sys_vec[i]->Dynamics(fg);
    fg_sys[i] = fg;
  }
  MatrixXd L_fg_obs(1, 1), L_fg_sys1(1, 5), L_fg_sys2(1, 5);
  //  Quadrotor-obstacle collision pairs
  int idx = 0;
  for (int i = 0; i < num_sys_; ++i) {
    for (int j = 0; j < num_obs_; ++j, ++idx) {
      env.obs_cps[idx]->LieDerivatives(fg_sys[i], fg_obs, L_fg_sys1, L_fg_obs);
      const double h_dist =
          env.obs_cps[idx]->get_minimum_distance() - margin2_(idx);
      const double dist_cbf_lb = -kAlphaDistCbf * h_dist - L_fg_sys1(0, 0);
      // if (h_dist <= -margin2_(idx) / 1.0) use_backup_control = true;

      for (int k = 0; k < 4; ++k)
        constraint_mat_.coeffRef(idx, 4 * i + k) = L_fg_sys1(0, k + 1);
      constraint_lb_(idx) = dist_cbf_lb;
      // constraint_lb_(idx) = -kInf;
    }
  }
  //  Inter-quadrotor collision pairs
  for (int i = 0; i < num_sys_; ++i) {
    for (int j = i + 1; j < num_sys_; ++j, ++idx) {
      env.sys_cps[idx - num_obs_cps_]->LieDerivatives(fg_sys[i], fg_sys[j],
                                                      L_fg_sys1, L_fg_sys2);
      const double h_dist =
          env.sys_cps[idx - num_obs_cps_]->get_minimum_distance() -
          margin2_(idx);
      const double dist_cbf_lb =
          -kAlphaDistCbf * h_dist - L_fg_sys1(0, 0) - L_fg_sys2(0, 0);
      // if (h_dist <= -margin2_(idx) / 1.0) use_backup_control = true;

      for (int k = 0; k < 4; ++k) {
        constraint_mat_.coeffRef(idx, 4 * i + k) = L_fg_sys1(0, k + 1);
        constraint_mat_.coeffRef(idx, 4 * j + k) = L_fg_sys2(0, k + 1);
      }
      constraint_lb_(idx) = dist_cbf_lb;
      // constraint_lb_(idx) = -kInf;
    }
  }

  // Velocity bound CBF constraint
  VectorXd vel_cbf_mat(4);
  for (int i = 0; i < num_sys_; ++i, ++idx) {
    const auto v = env.sys_vec[i]->x().segment<3>(3);
    const auto R = env.sys_vec[i]->x().tail<9>().reshaped(3, 3);
    const double v_norm = std::sqrt(std::pow(kEps * max_vel_, 2) + v.dot(v));
    const double h_vel =
        (1 - kVel) * max_vel_ - kVel * v.dot(R.col(2)) - v_norm;
    const auto y_vel = kVel * R.col(2) + v / v_norm;
    vel_cbf_mat(0) = -y_vel.dot(R.col(2)) / mass_;
    vel_cbf_mat.tail<3>() = -kVel * e3_hat * R.transpose() * v;
    const double vel_cbf_lb = -kAlphaVelCbf * h_vel - y_vel.dot(kGravityVec);

    for (int j = 0; j < 4; ++j)
      constraint_mat_.coeffRef(idx, 4 * i + j) = vel_cbf_mat(j);
    constraint_lb_(idx) = vel_cbf_lb;
    // constraint_lb_(idx) = -kInf;
  }

  // Quadrotor angle bound CBF constraint
  VectorXd ang_cbf_mat = VectorXd::Zero(4);
  for (int i = 0; i < num_sys_; ++i, ++idx) {
    const auto R = env.sys_vec[i]->x().tail<9>().reshaped(3, 3);
    const double h_ang = R(2, 2) - cos_max_ang_;
    ang_cbf_mat.tail<3>() = e3_hat * R.transpose() * e3;
    const double ang_cbf_lb = -kAlphaAngCbf * h_ang;

    for (int j = 0; j < 4; ++j)
      constraint_mat_.coeffRef(idx, 4 * i + j) = ang_cbf_mat(j);
    constraint_lb_(idx) = ang_cbf_lb;
    // constraint_lb_(idx) = -kInf;
  }

  // std::cout << constraint_lb_ << std::endl;
  // std::cout << constraint_mat_ << std::endl;
  // std::cout << constraint_ub_ << std::endl;

  // Update QP
  solver_.updateGradient(gradient_);
  solver_.updateLinearConstraintsMatrix(constraint_mat_);
  solver_.updateLowerBound(constraint_lb_);

  // Solve QP;
  // If not solved to optimality, use reference control
  if ((!use_backup_control) &&
      (solver_.solveProblem() == OsqpEigen::ErrorExitFlag::NoError)) {
    const VectorXd solution = solver_.getSolution();
    u = solution;
  } else {
    const double kD = 5.0;
    const double kR = 6.0;
    const double max_acc = 3.0 / 4.0 * kGravity;
    VectorXd a_ref(3);
    MatrixXd Rd(3, 3);
    for (int i = 0; i < num_sys_; ++i) {
      const auto vi = env.sys_vec[i]->x().segment<3>(3);
      const auto Ri = env.sys_vec[i]->x().tail<9>().reshaped(3, 3);
      a_ref = -kD * vi;
      const double normi = a_ref.norm();
      if (normi > max_acc) a_ref = a_ref / normi * max_acc;
      RotationFromZVector(a_ref + kGravityVec, Rd);
      // Thrust input
      u(4 * i) = env.mass * Ri.col(2).dot(a_ref + kGravityVec);
      // Angular velocity input
      u.segment<3>(4 * i + 1) =
          So3PTrackingControl(Ri, Rd, VectorXd::Zero(3), {kR, 0.0});
    }
  }
}

}  // namespace
