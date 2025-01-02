#include <Eigen/Core>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

// Data types
#include "sccbf/data_types.h"
// Geometries
#include "quadrotor_corridor.h"
#include "quadrotor_downwash.h"
#include "quadrotor_shape.h"
#include "quadrotor_uncertainty.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/geometry/static_polytope.h"
#include "sccbf/transformation/minkowski.h"
// Systems
#include "sccbf/system/dynamical_system.h"
#include "sccbf/system/quadrotor.h"
#include "sccbf/system/quadrotor_reduced.h"
// LCP solver
#include "sccbf/lemke.h"
// Solver
#include "sccbf/collision/collision_pair.h"
#include "sccbf/collision/distance_solver.h"
#include "sccbf/solver_options.h"
// Utils
#include "sccbf/utils/control_utils.h"
#include "sccbf/utils/matrix_utils.h"

using namespace sccbf;

namespace {

const double kPi = static_cast<double>(EIGEN_PI);

struct Environment {
  // Collision pair for geometries
  std::shared_ptr<CollisionPair> cp;
  // System (reduced quadrotor model)
  std::shared_ptr<DynamicalSystem> sys;
  // Solver options
  std::shared_ptr<SolverOptions> opt;

  void set_state(const VectorXd& x, const VectorXd& u);

  void set_state(const Eigen::Vector3d& p, const Eigen::Vector3d& v,
                 const Eigen::Matrix3d& R);

  void UpdateSystemState(const VectorXd& u, double dt);
};

void Environment::set_state(const VectorXd& x, const VectorXd& u) {
  const int nx = sys->nx();
  const int nu = sys->nu();
  assert(x.rows() == nx);
  assert(u.rows() == nu);

  sys->set_x(x);
  VectorXd f(nx);
  MatrixXd g(nx, nu);
  sys->Dynamics(f, g);
  const auto dx = f + g * u;
  cp->get_set2()->set_states(x, dx);
}

void Environment::set_state(const Eigen::Vector3d& p, const Eigen::Vector3d& v,
                            const Eigen::Matrix3d& R) {
  assert((R.rows() == 3) && (R.cols() == 3));

  const int nx = sys->nx();
  const int nu = sys->nu();
  VectorXd x(nx), u(nu);

  x.head<3>() = p;
  x.segment<3>(3) = v;
  x.tail<9>() = R.reshaped(9, 1);
  u = VectorXd::Zero(nu);
  set_state(x, u);
}

void Environment::UpdateSystemState(const VectorXd& u, double dt) {
  const int nu = sys->nu();
  assert(u.rows() == nu);

  sys->IntegrateDynamics(u, dt);
}

std::shared_ptr<ConvexSet> GetObstacle() {
  const int nz = 3;
  const int nr = nz * 5;
  const VectorXd center = VectorXd::Zero(nz);
  const double in_radius = nz * 0.5;  // [m]
  const double margin = 0.05;         // [m]
  const double sc_modulus = 0.0;

  MatrixXd A(nr, nz);
  VectorXd b(nr);
  RandomPolytope(center, in_radius, A, b);
  return std::make_shared<StaticPolytope<3>>(A, b, center, margin, sc_modulus,
                                             true);
}

std::shared_ptr<ConvexSet> GetRobotSafeRegion(int type) {
  // Quadrotor shape set
  const double pow = 2.5;
  Eigen::Vector4d coeff(1.0, 1.0, 0.4, 0.3);  // [m, m, m, ]
  double margin = 0.05;                       // [m]
  const std::shared_ptr<ConvexSet> shape =
      std::make_shared<QuadrotorShape>(pow, coeff, margin);

  // Quadrotor uncertainty set
  MatrixXd Q(3, 3);    // [1/m^2].
  Q << 3.0, 1.0, 0.0,  //
      1.0, 2.0, 0.0,   //
      0.0, 0.0, 1.0;
  coeff << 1.0, 0.5, 1.0, 0.0;  // [, , 1/m, ]
  margin = 0.0;
  const std::shared_ptr<ConvexSet> uncertainty =
      std::make_shared<QuadrotorUncertainty>(Q, coeff, margin);

  // Quadrotor corridor set
  const double stop_time = 2.0;         // [s]
  const double orientation_cost = 1.0;  // [s]
  const double max_vel = 1.0;           // [m/s]
  margin = 0.0;                         // [m]
  const std::shared_ptr<ConvexSet> corridor =
      std::make_shared<QuadrotorCorridor>(stop_time, orientation_cost, max_vel,
                                          margin);

  // Quadrotor downwash set
  MatrixXd A(5, 3);    // [1/m]
  A << 4.0, 0.0, 2.0,  //
      0.0, 4.0, 2.0,   //
      -4.0, 0.0, 2.0,  //
      0.0, -4.0, 2.0,  //
      0.0, 0.0, -1.5;
  const VectorXd b = VectorXd::Zero(5);  // []
  const double level = 1.5;              // []
  margin = 0.05;                         // [m]
  const std::shared_ptr<ConvexSet> downwash =
      std::make_shared<QuadrotorDownwash>(A, b, level, margin);

  switch (type) {
    // shape + uncertainty
    case 0:
      return std::make_shared<MinkowskiSumSet>(shape, uncertainty);
    // shape + corridor
    case 1:
      return std::make_shared<MinkowskiSumSet>(shape, corridor);
    // downwash + corridor
    case 2:
      return std::make_shared<MinkowskiSumSet>(downwash, corridor);

    default:
      throw std::runtime_error("type should be 0, 1, or 2!");
  }
}

void SetupEnvironment(int type, double dt, Environment& env) {
  // Set random seed
  std::srand(10);

  // Set obstacle (static polytope)
  const auto c1 = GetObstacle();
  VectorXd x = VectorXd::Zero(0);
  c1->set_states(x, x);
  // Set quadrotor safe region (depending on type)
  const auto c2 = GetRobotSafeRegion(type);

  // Set solver options and distance solver
  auto solver = std::make_shared<DistanceSolver>();
  MatrixXd metric = MatrixXd::Identity(3, 3);
  auto opt = std::make_shared<SolverOptions>();
  opt->metric = metric;
  opt->kkt_ode.use_kkt_err_tol = false;
  opt->kkt_ode.timestep = dt;

  // Set collision pair
  auto cp = std::make_shared<CollisionPair>(c1, c2, opt, solver);

  // Set dynamical system
  const double mass = 0.5;  // [kg].
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);
  std::shared_ptr<DynamicalSystem> sys =
      std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u);

  // Update environment struct
  env.cp = cp;
  env.sys = sys;
  env.opt = opt;
}

struct Trajectory {
  double time;
  Eigen::Vector3d pd;
  Eigen::Vector3d vd;
  Eigen::Vector3d ad;
  Eigen::Vector3d wd;
};

void GetDesiredTrajectory(double t, Trajectory& trajectory) {
  assert(t >= 0);

  const double scale = 2.0;
  const double radius = 7.0;                           // [m]
  const double T_xy = 10.0 * scale;                    // [s]
  const double height = 3.0;                           // [m]
  const double T_z = 11.0 * scale;                     // [s]
  const Eigen::Vector3d w(0.0, 0.0, 2.0 * kPi / 9.0);  // [rad/s]

  Eigen::Vector3d p, v, a;
  const double f_xy = 2.0 * kPi / T_xy;
  const double f_z = 2.0 * kPi / T_z;
  p << radius * std::cos(f_xy * t), radius * std::sin(f_xy * t),
      height * std::sin(f_z * t);
  v << -radius * f_xy * std::sin(f_xy * t), radius * f_xy * std::cos(f_xy * t),
      height * f_z * std::cos(f_z * t);
  a << -radius * f_xy * f_xy * std::cos(f_xy * t),
      -radius * f_xy * f_xy * std::sin(f_xy * t),
      -height * f_z * f_z * std::sin(f_z * t);

  trajectory.time = t;
  trajectory.pd = p;
  trajectory.vd = v;
  trajectory.ad = a;
  trajectory.wd = w;
}

void GetTrackingControl(const std::shared_ptr<DynamicalSystem>& sys,
                        const Trajectory& trajectory, VectorXd& u) {
  const int nu = sys->nu();
  assert(u.rows() == nu);

  const VectorXd x = sys->x();
  const auto p = x.head<3>();
  const auto v = x.segment<3>(3);
  const auto R = x.segment<9>(6).reshaped(3, 3);
  const auto pd = trajectory.pd;
  const auto vd = trajectory.vd;
  const auto ad = trajectory.ad;
  const auto wd = trajectory.wd;

  const double k_p = 1.0;
  const double k_d = 1.0;
  const double k_R = 5.0;
  const double mass = 0.5;                  // [kg]
  const Eigen::Vector3d g(0.0, 0.0, 9.81);  // [m/s^2]
  const double max_acc = 3.0 / 4.0 * g.norm();

  VectorXd a_ref = ad - k_p * (p - pd) - k_d * (v - vd);
  const double norm = a_ref.norm();
  if (norm > max_acc) a_ref = a_ref / norm * max_acc;
  MatrixXd Rd(3, 3);
  RotationFromZVector(a_ref + g, Rd);
  // Thrust input
  u(0) = mass * R.col(2).dot(a_ref + g);
  // Angular velocity input
  u.tail<3>() = So3PTrackingControl(R, Rd, wd, {k_R, 0.0});
}

struct Logs {
  VectorXd t;  // (t_seq, t_0, T, dt)
  MatrixXd x;
  VectorXd solve_time_ode;
  VectorXd solve_time_opt;
  VectorXd dist2_ode;
  VectorXd dist2_opt;
  VectorXd Ddist2_ode;
  MatrixXd z_ode_;
  MatrixXd z_opt_;
  VectorXd z_err_norm;
  VectorXd z_opt_norm;
  MatrixXd lambda_ode_;
  MatrixXd lambda_opt_;
  VectorXd lambda_err_norm;
  VectorXd lambda_opt_norm;
  VectorXd dual_inf_err_norm;
  VectorXd prim_inf_err_norm;
  VectorXd compl_err_norm;
  int num_opt_solves;

  Logs(int Nlog, int nx, int nz, int nr)
      : t(Nlog),
        x(nx, Nlog),
        solve_time_ode(Nlog),
        solve_time_opt(Nlog),
        dist2_ode(Nlog),
        dist2_opt(Nlog),
        Ddist2_ode(Nlog),
        z_ode_(nz, Nlog),
        z_opt_(nz, Nlog),
        z_err_norm(Nlog),
        z_opt_norm(Nlog),
        lambda_ode_(nr, Nlog),
        lambda_opt_(nr, Nlog),
        lambda_err_norm(Nlog),
        lambda_opt_norm(Nlog),
        dual_inf_err_norm(Nlog),
        prim_inf_err_norm(Nlog),
        compl_err_norm(Nlog),
        num_opt_solves{0} {}

  void UpdateOdeLogs(int i, const Environment& env, double kkt_err,
                     double solve_time) {
    const int Nlog = static_cast<int>(t.rows());
    assert((i >= 0) && (i < Nlog));
    const int nz = static_cast<int>(z_ode_.rows());
    const int nr = static_cast<int>(lambda_ode_.rows());
    VectorXd z(nz), lambda(nr);
    VectorXd dual_inf_err(nz), prim_inf_err(nr), compl_err(nr);

    x.col(i) = env.sys->x();
    solve_time_ode(i) = solve_time;
    dist2_ode(i) = env.cp->get_kkt_solution(z, lambda);
    Ddist2_ode(i) = env.cp->MinimumDistanceDerivative();
    z_ode_.col(i) = z;
    lambda_ode_.col(i) = lambda;
    env.cp->KktError(dual_inf_err, prim_inf_err, compl_err);
    dual_inf_err_norm(i) = dual_inf_err.lpNorm<Eigen::Infinity>();
    prim_inf_err_norm(i) = prim_inf_err.lpNorm<Eigen::Infinity>();
    compl_err_norm(i) = compl_err.lpNorm<Eigen::Infinity>();

    if (env.opt->kkt_ode.use_kkt_err_tol)
      num_opt_solves += (kkt_err > env.opt->kkt_ode.max_inf_kkt_err);
    else
      num_opt_solves += (kkt_err > env.opt->kkt_ode.max_primal_dual_gap);
  }

  void UpdateOptLogs(int i, const Environment& env, double solve_time) {
    const int Nlog = static_cast<int>(t.rows());
    assert((i >= 0) && (i < Nlog));
    const int nz = static_cast<int>(z_ode_.rows());
    const int nr = static_cast<int>(lambda_ode_.rows());
    VectorXd z(nz), lambda(nr);

    solve_time_opt(i) = solve_time;
    dist2_opt(i) = env.cp->get_kkt_solution(z, lambda);
    z_opt_.col(i) = z;
    z_opt_norm(i) = z.norm();
    lambda_opt_.col(i) = lambda;
    lambda_opt_norm(i) = lambda.norm();
  }

  void ComputeErrorNorms() {
    const int Nlog = static_cast<int>(t.rows());

    for (int i = 0; i < Nlog; ++i) {
      z_err_norm(i) = (z_ode_.col(i) - z_opt_.col(i)).norm();
      lambda_err_norm(i) = (lambda_ode_.col(i) - lambda_opt_.col(i)).norm();
    }
  }

  void SaveLogs(std::ofstream& outfile) {
    const int Nlog = static_cast<int>(t.rows());
    const int nx = static_cast<int>(x.rows());

    // Header
    outfile << "t (s),";
    for (int i = 0; i < nx; ++i) outfile << "x_" << i << ",";
    outfile << "solve time (ode) (s),solve time (opt) (s),"
            << "dist2 (ode) (m^2),dist2 (opt) (m^2),"
            << "D(dist2) (ode) (m^2/s),"
            << "|z_opt - z_ode|,|z_opt|,|lambda_opt - lambda_ode|,|lambda_opt|,"
            << "dual inf err,primal inf err,complementarity err,"
            << "#ipopt solves=," << num_opt_solves << std::endl;

    // Data
    for (int k = 0; k < Nlog; ++k) {
      outfile << t(k) << ",";
      for (int i = 0; i < nx; ++i) outfile << x(i, k) << ",";
      outfile << solve_time_ode(k) << "," << solve_time_opt(k) << ","
              << dist2_ode(k) << "," << dist2_opt(k) << "," << Ddist2_ode(k)
              << "," << z_err_norm(k) << "," << z_opt_norm(k) << ","
              << lambda_err_norm(k) << "," << lambda_opt_norm(k) << ","
              << dual_inf_err_norm(k) << "," << prim_inf_err_norm(k) << ","
              << compl_err_norm(k) << std::endl;
    }
  }
};

}  // namespace

int main() {
  // Set time sequences
  const double t_0 = 0.0;  // [s]
  const double T = 50.0;   // [s]
  double dt = 1e-3;
  const int N = static_cast<int>(std::ceil(T / dt));
  const auto t_seq = VectorXd::LinSpaced(N, t_0, T);
  dt = t_seq(1) - t_seq(0);  // [s]
  const int log_freq = 10;
  const int Nlog = static_cast<int>(std::ceil(N / log_freq));

  Trajectory trajectory;

  // Get environment, and initialize states and minimum distance
  Environment env;
  const int type = 0;
  SetupEnvironment(type, dt, env);
  GetDesiredTrajectory(t_0, trajectory);
  env.set_state(trajectory.pd, trajectory.vd, Eigen::Matrix3d::Identity());
  env.cp->MinimumDistance();

  // Initialize logs
  const int nx = env.sys->nx();
  const int nu = env.sys->nu();
  const int nz = env.cp->get_set1()->nz() + env.cp->get_set2()->nz();
  const int nr = env.cp->get_set1()->nr() + env.cp->get_set2()->nr();
  Logs log(Nlog, nx, nz, nr);
  log.t = VectorXd::LinSpaced(Nlog, t_0, T);

  // KKT ODE solution
  auto start = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  VectorXd u(nu);
  double kkt_err = 0.0;
  double solve_time = 0.0;
  printf("Solving KKT ODE...\n");
  for (int i = 0, j = 0; i < N; ++i) {
    // Get desired trajectory
    GetDesiredTrajectory(t_seq(i), trajectory);
    // Get tracking input
    GetTrackingControl(env.sys, trajectory, u);
    // Update environment states
    env.set_state(env.sys->x(), u);
    // Update KKT solutions
    start = std::chrono::high_resolution_clock::now();
    kkt_err = env.cp->KktStep();
    elapsed = std::chrono::high_resolution_clock::now() - start;
    solve_time =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
                .count()) *
        1e-6;
    // Log data
    if (i % log_freq == 0) {
      log.UpdateOdeLogs(j, env, kkt_err, solve_time);
      ++j;
    }
    // Integrate state
    env.UpdateSystemState(u, dt);
  }
  printf("KKT ODE solution computed.\n");

  // Optimization solution
  u = VectorXd::Zero(nu);
  printf("Solving minimum distance optimization problems...\n");
  for (int i = 0; i < Nlog; ++i) {
    // Set environment state
    env.set_state(log.x.col(i), u);
    // Solve minimum distance optimization
    start = std::chrono::high_resolution_clock::now();
    env.cp->MinimumDistance();
    elapsed = std::chrono::high_resolution_clock::now() - start;
    solve_time =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
                .count()) *
        1e-6;
    // Update logs
    log.UpdateOptLogs(i, env, solve_time);
  }
  printf("Minimum distance optimization problems solved.\n");

  log.ComputeErrorNorms();

  // Save logs to .csv file
  printf("Saving logs...\n");
  std::string filename = "kkt_ode_data_" + std::to_string(type) + ".csv";
  std::ofstream outfile(filename);
  log.SaveLogs(outfile);
  outfile.close();
  printf("Logs saved.\n");

  return EXIT_SUCCESS;
}
