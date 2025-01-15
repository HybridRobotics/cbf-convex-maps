#include <Eigen/Core>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ma_cbf_example/controller.h"

// Data types
#include "sccbf/data_types.h"
// Geometries
#include "sccbf/geometry/convex_set.h"
#include "sccbf/geometry/quadrotor_corridor.h"
#include "sccbf/geometry/quadrotor_downwash.h"
#include "sccbf/geometry/quadrotor_shape.h"
#include "sccbf/geometry/static_polytope.h"
#include "sccbf/transformation/minkowski.h"
// Systems
#include "sccbf/system/dynamical_system.h"
#include "sccbf/system/quadrotor.h"
#include "sccbf/system/quadrotor_reduced.h"
// Solver
#include "sccbf/collision/collision_pair.h"
#include "sccbf/collision/distance_solver.h"
#include "sccbf/solver_options.h"
// Utils
#include "sccbf/utils/control_utils.h"
#include "sccbf/utils/matrix_utils.h"

using namespace sccbf;

namespace {

void GetObstacle(std::vector<std::shared_ptr<ConvexSet>>& obs_vec,
                 MatrixXd& centers) {
  obs_vec.resize(0);
  centers.resize(3, 0);

  MatrixXd A;
  VectorXd b, p(3);
  double margin = 0.1;  // [m]
  double sc_modulus = 1e-2;

  // Dodecahedron polytope
  double a1 = 0.52573;
  double a2 = 0.85065;
  double a3 = 1.37638;
  A.resize(12, 3);
  A << a1, a2, 0.0,   //
      a1, -a2, 0.0,   //
      -a1, a2, 0.0,   //
      -a1, -a2, 0.0,  //
      0.0, a1, a2,    //
      0.0, a1, -a2,   //
      0.0, -a1, a2,   //
      0.0, -a1, -a2,  //
      a2, 0.0, a1,    //
      -a2, 0.0, a1,   //
      a2, 0.0, -a1,   //
      -a2, 0.0, -a1;
  b = a3 * VectorXd::Ones(12);

  // Obstacles
  p << -7.0, 1.5, -0.0;
  centers.conservativeResize(3, centers.cols() + 1);
  obs_vec.push_back(
      std::make_shared<StaticPolytope<3>>(A, b, p, margin, sc_modulus, true));
  centers.col(centers.cols() - 1) = p;

  p << -6.0, -1.5, -2.0;
  centers.conservativeResize(3, centers.cols() + 1);
  obs_vec.push_back(
      std::make_shared<StaticPolytope<3>>(A, b, p, margin, sc_modulus, true));
  centers.col(centers.cols() - 1) = p;

  p << -3.5, 1.5, -2.5;
  centers.conservativeResize(3, centers.cols() + 1);
  obs_vec.push_back(
      std::make_shared<StaticPolytope<3>>(A, b, p, margin, sc_modulus, true));
  centers.col(centers.cols() - 1) = p;

  p << -0.0, 1.0, -1.5;
  centers.conservativeResize(3, centers.cols() + 1);
  obs_vec.push_back(
      std::make_shared<StaticPolytope<3>>(A, b, p, margin, sc_modulus, true));
  centers.col(centers.cols() - 1) = p;

  p << 1.0, -1.0, -1.5;
  centers.conservativeResize(3, centers.cols() + 1);
  obs_vec.push_back(
      std::make_shared<StaticPolytope<3>>(A, b, p, margin, sc_modulus, true));
  centers.col(centers.cols() - 1) = p;

  p << 4.0, 1.0, 1.0;
  centers.conservativeResize(3, centers.cols() + 1);
  obs_vec.push_back(
      std::make_shared<StaticPolytope<3>>(A, b, p, margin, sc_modulus, true));
  centers.col(centers.cols() - 1) = p;

  p << 8.0, 0.0, -1.5;
  centers.conservativeResize(3, centers.cols() + 1);
  obs_vec.push_back(
      std::make_shared<StaticPolytope<3>>(A, b, p, margin, sc_modulus, true));
  centers.col(centers.cols() - 1) = p;

  // Wall 1
  A.resize(1, 3);
  A << 0.0, 1.0, 0.0;
  b.resize(1);
  b << -3.0;
  p << 0.0, 0.0, 0.0;
  obs_vec.push_back(
      std::make_shared<StaticPolytope<3>>(A, b, p, margin, 0.0, true));

  // Wall 2
  A.resize(1, 3);
  A << 0.0, -1.0, 0.0;
  b.resize(1);
  b << -3.0;
  p << 0.0, 0.0, 0.0;
  obs_vec.push_back(
      std::make_shared<StaticPolytope<3>>(A, b, p, margin, 0.0, true));

  // Wall 3
  A.resize(1, 3);
  A << 0.0, 0.0, 1.0;
  b.resize(1);
  b << -3.0;
  p << 0.0, 0.0, 0.0;
  obs_vec.push_back(
      std::make_shared<StaticPolytope<3>>(A, b, p, margin, 0.0, true));

  // Wall 4
  A.resize(1, 3);
  A << 0.0, 0.0, -1.0;
  b.resize(1);
  b << -3.0;
  p << 0.0, 0.0, 0.0;
  obs_vec.push_back(
      std::make_shared<StaticPolytope<3>>(A, b, p, margin, 0.0, true));
}

void GetRobotSafeRegion(int num_sys,
                        std::vector<std::shared_ptr<ConvexSet>>& safe_vec,
                        double max_vel) {
  safe_vec.resize(0);

  // Quadrotor shape set parameters
  const double pow = 2.5;
  Eigen::Vector4d coeff(1.0, 1.0, 0.4, 0.3);  // [m, m, m, ]
  const double margin_s = 0.05;               // [m]
  // Quadrotor corridor set parameters
  const double stop_time = 1.0;         // [s]
  const double orientation_cost = 0.5;  // [s]
  const double margin_c = 0.05;         // [m]

  for (int i = 0; i < num_sys; ++i) {
    const std::shared_ptr<ConvexSet> shape =
        std::make_shared<QuadrotorShape>(pow, coeff, margin_s);
    const std::shared_ptr<ConvexSet> corridor =
        std::make_shared<QuadrotorCorridor>(stop_time, orientation_cost,
                                            max_vel, margin_c);
    safe_vec.push_back(std::make_shared<MinkowskiSumSet>(shape, corridor));
  }
}

void SetupEnvironment(double dt, Environment& env) {
  // Get obstacles (static polytopes)
  std::vector<std::shared_ptr<ConvexSet>> obs_vec;
  MatrixXd centers;
  GetObstacle(obs_vec, centers);
  int num_obs = static_cast<int>(obs_vec.size());
  VectorXd x = VectorXd::Zero(0);
  for (auto obs : obs_vec) obs->set_states(x, x);
  // Get quadrotor safe regions
  const int num_sys = 1;
  const double max_vel = 0.5;  // [m/s]
  std::vector<std::shared_ptr<ConvexSet>> safe_vec;
  GetRobotSafeRegion(num_sys, safe_vec, max_vel);

  // Set solver options and distance solver
  auto solver = std::make_shared<DistanceSolver>();
  MatrixXd metric = MatrixXd::Identity(3, 3);
  auto opt = std::make_shared<SolverOptions>();
  opt->metric = metric;
  opt->kkt_ode.use_kkt_err_tol = true;
  opt->kkt_ode.max_primal_dual_gap = 10e-3;  // [m]
  opt->kkt_ode.max_inf_kkt_err = 1e-1;
  opt->kkt_ode.timestep = dt;

  // Set collision pairs
  //  Quadrotor-obstalce collision pairs
  std::vector<std::shared_ptr<CollisionPair>> obs_cps;
  for (int i = 0; i < num_sys; ++i)
    for (int j = 0; j < num_obs; ++j)
      obs_cps.push_back(std::make_shared<CollisionPair>(safe_vec[i], obs_vec[j],
                                                        opt, solver));
  //  Inter quadrotor collision pairs
  std::vector<std::shared_ptr<CollisionPair>> sys_cps;
  for (int i = 0; i < num_sys; ++i)
    for (int j = i + 1; j < num_sys; ++j)
      sys_cps.push_back(std::make_shared<CollisionPair>(
          safe_vec[i], safe_vec[j], opt, solver));

  // Set dynamical systems
  const double mass = 0.5;  // [kg].
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);
  std::vector<std::shared_ptr<DynamicalSystem>> sys_vec;
  for (int i = 0; i < num_sys; ++i)
    sys_vec.push_back(
        std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u));

  // Update environment struct
  env.num_obs = num_obs;
  env.centers = centers;
  env.obs_cps = obs_cps;
  env.num_sys = num_sys;
  env.sys_vec = sys_vec;
  env.sys_cps = sys_cps;

  env.mass = mass;
  env.max_vel = max_vel;

  env.opt = opt;
}

struct Trajectory {
  double time;
  Eigen::Vector3d pd;
  Eigen::Vector3d vd;
  Eigen::Vector3d ad;
  Eigen::Vector3d wd;
};

void GetDesiredTrajectory(double t, int i, const Environment& env,
                          Trajectory& trajectory) {
  assert(t >= 0);
  assert((i >= 0) && (i < 16));

  const double side_len = 1.0;     // [m]
  const double separation = 12.0;  // [m]

  Eigen::Vector3d p;
  const Eigen::Vector3d v(0.5, 0.0, 0.0);  // [m/s]
  const Eigen::Vector3d a(0.0, 0.0, 0.0);  // [m/s^2]
  const Eigen::Vector3d w(0.0, 0.0, 0.0);  // [rad/s]

  const int b0 = i % 2;        // team index
  const int b1 = (i / 4) % 2;  // x index
  const int b2 = (i / 2) % 2;  // y index
  const int b3 = (i / 8) % 2;  // z index

  p(0) = std::pow(-1, b1) * side_len - separation;
  p(1) = std::pow(-1, b2) * side_len - 1.0;
  p(2) = std::pow(-1, b3) * side_len - 1.0;
  p(0) = std::pow(-1, b0) * p(0);
  if (t > 0.05) p(0) = std::min(separation, env.sys_vec[i]->x()(0) + 2.0);
  // if (t > 0.05) p = -p;

  trajectory.time = t;
  trajectory.pd = p;
  trajectory.vd = v;
  trajectory.ad = a;
  trajectory.wd = w;
}

void GetReferenceControl(double t, const Environment& env, VectorXd& u) {
  const int num_sys = env.num_sys;
  assert(u.rows() == 4 * num_sys);

  const double kP = 10.0;
  const double kD = 5.0;
  const double kR = 6.0;
  const Eigen::Vector3d g(0.0, 0.0, 9.81);  // [m/s^2]
  const double max_acc = 3.0 / 4.0 * g(2);

  Trajectory trajectory;
  MatrixXd Rd(3, 3);
  for (int i = 0; i < num_sys; ++i) {
    const VectorXd xi = env.sys_vec[i]->x();
    const auto pi = xi.head<3>();
    const auto vi = xi.segment<3>(3);
    const auto Ri = xi.segment<9>(6).reshaped(3, 3);
    GetDesiredTrajectory(t, i, env, trajectory);
    const auto pdi = trajectory.pd;
    const auto vdi = trajectory.vd;
    const auto adi = trajectory.ad;
    const auto wdi = trajectory.wd;

    VectorXd a_refi = adi - kP * (pi - pdi) - kD * (vi - vdi);
    const double normi = a_refi.norm();
    if (normi > max_acc) a_refi = a_refi / normi * max_acc;
    RotationFromZVector(a_refi + g, Rd);
    // Thrust input
    u(4 * i) = env.mass * Ri.col(2).dot(a_refi + g);
    // Angular velocity input
    u.segment<3>(4 * i + 1) = So3PTrackingControl(Ri, Rd, wdi, {kR, 0.0});
  }
}

struct Logs {
  VectorXd t;  // (t_seq, t_0, T, dt)
  int num_obs;
  int num_sys;
  int num_obs_cps;
  int num_sys_cps;
  int num_cps;
  MatrixXd x;
  VectorXd solve_time_ode;
  VectorXd solve_time_qp;
  VectorXd solve_time_opt;
  MatrixXd dist2_ode;
  MatrixXd dist2_opt;
  MatrixXd Ddist2_ode;
  MatrixXd z_err_norm;
  MatrixXd z_opt_norm;
  MatrixXd lambda_err_norm;
  MatrixXd lambda_opt_norm;
  int num_opt_solves;

  Logs(int Nlog, int num_sys, int num_cps, int nx)
      : t(Nlog),
        num_cps(num_cps),
        x(num_sys * nx, Nlog),
        solve_time_ode(Nlog),
        solve_time_qp(Nlog),
        solve_time_opt(Nlog),
        dist2_ode(num_cps, Nlog),
        dist2_opt(num_cps, Nlog),
        Ddist2_ode(num_cps, Nlog),
        z_err_norm(num_cps, Nlog),
        z_opt_norm(num_cps, Nlog),
        lambda_err_norm(num_cps, Nlog),
        lambda_opt_norm(num_cps, Nlog),
        num_opt_solves{0} {}

  void UpdateLogs(int k, const Environment& env, double solve_time_ode,
                  double solve_time_qp, const VectorXd& kkt_err);

  void SaveLogs(std::ofstream& outfile);
};

void Logs::UpdateLogs(int k, const Environment& env, double solve_time_ode,
                      double solve_time_qp, const VectorXd& kkt_err) {
  const int Nlog = static_cast<int>(t.rows());
  assert((k >= 0) && (k < Nlog));

  for (int i = 0; i < num_sys; ++i)
    x.block<15, 1>(15 * i, k) = env.sys_vec[i]->x();
  this->solve_time_ode(k) = solve_time_ode;
  this->solve_time_qp(k) = solve_time_qp;

  VectorXd z_ode, z_opt, lambda_ode, lambda_opt;
  double solve_time = 0.0;

  auto kkt_err_function = [this, &env, &kkt_err, &z_ode, &z_opt, &lambda_ode,
                           &lambda_opt,
                           &solve_time](int idx, int k,
                                        std::shared_ptr<CollisionPair> cp) {
    int nz = cp->get_set1()->nz() + cp->get_set2()->nz();
    int nr = cp->get_set1()->nr() + cp->get_set2()->nr();
    z_ode.resize(nz);
    z_opt.resize(nz);
    lambda_ode.resize(nr);
    lambda_opt.resize(nr);
    // Store current KKT solution
    this->dist2_ode(idx, k) = cp->get_kkt_solution(z_ode, lambda_ode);
    this->Ddist2_ode(idx, k) = cp->GetMinimumDistanceDerivative();
    // Get ipopt solution
    auto start = std::chrono::high_resolution_clock::now();
    cp->MinimumDistance();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    solve_time +=
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
                .count()) *
        1e-6;
    this->dist2_opt(idx, k) = cp->get_kkt_solution(z_opt, lambda_opt);
    // Restore KKT solution
    cp->set_kkt_solution(this->dist2_ode(idx, k), z_ode, lambda_ode);

    // Error norms
    z_err_norm(idx, k) = (z_ode - z_opt).norm();
    z_opt_norm(idx, k) = z_opt.norm();
    lambda_err_norm(idx, k) = (lambda_ode - lambda_opt).norm();
    lambda_opt_norm(idx, k) = lambda_opt.norm();

    // num_opt_solves
    if (env.opt->kkt_ode.use_kkt_err_tol)
      this->num_opt_solves += (kkt_err(idx) > env.opt->kkt_ode.max_inf_kkt_err);
    else
      this->num_opt_solves +=
          (kkt_err(idx) > env.opt->kkt_ode.max_primal_dual_gap);
  };

  int idx = 0;
  for (int i = 0; i < num_obs_cps; ++i, ++idx)
    kkt_err_function(idx, k, env.obs_cps[i]);
  for (int i = 0; i < num_sys_cps; ++i, ++idx)
    kkt_err_function(idx, k, env.sys_cps[i]);

  solve_time_opt(k) = solve_time;
}

void Logs::SaveLogs(std::ofstream& outfile) {
  const int Nlog = static_cast<int>(t.rows());

  // Header
  outfile << "#obstacles,#systems,#obstacle CPs,#system CPs,#CPs,#ipopt solves"
          << std::endl;
  outfile << num_obs << "," << num_sys << "," << num_obs_cps << ","
          << num_sys_cps << "," << num_cps << "," << num_opt_solves
          << std::endl;
  outfile << "t (s),";
  for (int i = 0; i < num_sys; ++i)
    for (int j = 0; j < 15; ++j) outfile << i << "_x_" << j << ",";
  outfile << "solve time (ode) (s),solve time (opt) (s),solve time (qp) (s),"
          << "dist2 (ode) (m^2)" << std::string(num_cps, ',')
          << "dist2 (opt) (m^2)" << std::string(num_cps, ',')
          << "D(dist2) (ode) (m^2/s)" << std::string(num_cps, ',')
          << "|z_opt - z_ode|" << std::string(num_cps, ',') << "|z_opt|"
          << std::string(num_cps, ',') << "|lambda_opt - lambda_ode|"
          << std::string(num_cps, ',') << "|lambda_opt|"
          << std::string(num_cps - 1, ',') << std::endl;

  // Data
  for (int k = 0; k < Nlog; ++k) {
    outfile << t(k) << ",";
    for (int i = 0; i < num_sys; ++i)
      for (int j = 0; j < 15; ++j) outfile << x(15 * i + j, k) << ",";
    outfile << solve_time_ode(k) << "," << solve_time_opt(k) << ","
            << solve_time_qp(k) << ",";
    for (int i = 0; i < num_cps; ++i) outfile << dist2_ode(i, k) << ",";
    for (int i = 0; i < num_cps; ++i) outfile << dist2_opt(i, k) << ",";
    for (int i = 0; i < num_cps; ++i) outfile << Ddist2_ode(i, k) << ",";
    for (int i = 0; i < num_cps; ++i) outfile << z_err_norm(i, k) << ",";
    for (int i = 0; i < num_cps; ++i) outfile << z_opt_norm(i, k) << ",";
    for (int i = 0; i < num_cps; ++i) outfile << lambda_err_norm(i, k) << ",";
    for (int i = 0; i < num_cps - 1; ++i)
      outfile << lambda_opt_norm(i, k) << ",";
    outfile << lambda_opt_norm(num_cps - 1, k) << std::endl;
  }
}

}  // namespace

int main() {
  // Set time sequences
  const double t_0 = 0.0;  // [s]
  const double T = 100.0;  // [s]
  double dt = 1e-3;
  const int N = static_cast<int>(std::ceil(T / dt));
  const auto t_seq = VectorXd::LinSpaced(N, t_0, T);
  dt = t_seq(1) - t_seq(0);  // [s]
  const int log_freq = 10;
  const int Nlog = static_cast<int>(std::ceil(N / log_freq));
  const int control_freq = log_freq;

  Trajectory trajectory;

  // Get environment, and initialize states and minimum distance
  Environment env;
  SetupEnvironment(dt, env);
  const int num_sys = env.num_sys;
  for (int i = 0; i < num_sys; ++i) {
    GetDesiredTrajectory(t_0, i, env, trajectory);
    env.set_state(i, trajectory.pd, trajectory.vd, Eigen::Matrix3d::Identity());
  }
  for (auto cp : env.obs_cps) cp->MinimumDistance();
  for (auto cp : env.sys_cps) cp->MinimumDistance();
  const int num_obs_cps = static_cast<int>(env.obs_cps.size());
  const int num_sys_cps = static_cast<int>(env.sys_cps.size());
  const int num_cps = num_obs_cps + num_sys_cps;

  // Initialize logs
  Logs log(Nlog, num_sys, num_cps, 15);
  log.t = VectorXd::LinSpaced(Nlog, t_0, T);
  log.num_obs = env.num_obs;
  log.num_sys = num_sys;
  log.num_obs_cps = num_obs_cps;
  log.num_sys_cps = num_sys_cps;

  // Control loop
  CbfQpController controller(env);
  VectorXd u_f = VectorXd::Zero(4 * num_sys);
  for (int i = 0; i < num_sys; ++i) u_f(4 * i) = env.mass * 9.81;
  const double k_f = 1.0;

  auto start = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  VectorXd u_ref(4 * num_sys), u(4 * num_sys);
  VectorXd kkt_err(num_cps);
  double solve_time_ode = 0.0;
  double solve_time_qp = 0.0;
  printf("Entering control loop...\n");
  for (int i = 0, j = 0; i < N; ++i) {
    if (i % 5000 == 0) {
      printf("time (s) = %5.2f\n", t_seq(i));
    }
    // if (i % control_freq == 0) {
    // Get reference input
    GetReferenceControl(t_seq(i), env, u_ref);
    // Solve CBF-QP
    start = std::chrono::high_resolution_clock::now();
    controller.Control(env, u_ref, u);
    // u = u_ref;
    elapsed = std::chrono::high_resolution_clock::now() - start;
    solve_time_qp =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
                .count()) *
        1e-6;
    // }
    // Filter inputs
    u_f = k_f * u + (1 - k_f) * u_f;
    for (int j = 0; j < num_sys; ++j) u_f(4 * j) = u(4 * j);
    // Update KKT solutions
    env.set_inputs(u_f);
    start = std::chrono::high_resolution_clock::now();
    int idx = 0;
    for (int k = 0; k < num_obs_cps; ++k, ++idx) {
      // const auto p = env.obs_cps[k]->get_set1()->x().head<3>();
      // if ((k < env.centers.cols()) && (p - env.centers.col(k)).norm() > 5.0)
      //   continue;
      kkt_err(idx) = env.obs_cps[k]->KktStep();
    }
    for (int k = 0; k < num_sys_cps; ++k, ++idx)
      kkt_err(idx) = env.sys_cps[k]->KktStep();
    elapsed = std::chrono::high_resolution_clock::now() - start;
    solve_time_ode =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
                .count()) *
        1e-6;
    // Compute ipopt solution and log data
    if (i % log_freq == 0) {
      log.UpdateLogs(j, env, solve_time_ode, solve_time_qp, kkt_err);
      ++j;
    }
    // Integrate state
    env.UpdateSystemState(u_f, dt);
  }
  printf("Control loop finished.\n");

  // Save logs to .csv file
  printf("Saving logs...\n");
  std::string filename = "ma_cbf_data.csv";
  std::ofstream outfile(filename);
  log.SaveLogs(outfile);
  outfile.close();
  printf("Logs saved.\n");

  return EXIT_SUCCESS;
}
