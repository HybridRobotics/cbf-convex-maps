#include <Eigen/Core>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cbf_example/controller.h"

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

// TODO: Set up a more complex scene
void GetObstacle(std::vector<std::shared_ptr<ConvexSet>>& obs_vec) {
  const int nz = 3;
  const int nr = nz * 5;
  const VectorXd center = VectorXd::Zero(nz);
  const double in_radius = nz * 0.5;  // [m]
  const double half_side_len = 3.0;   // [m]
  const double margin = 0.05;         // [m]
  const double sc_modulus = 0.0;

  MatrixXd A_(nr, nz);
  VectorXd b_(nr);
  RandomPolytope(center, in_radius, A_, b_);
  MatrixXd A(nr + 6, nz);
  VectorXd b(nr + 6);
  A.topRows(nr) = A_;
  A.middleRows<3>(nr) = MatrixXd::Identity(3, 3);
  A.bottomRows<3>() = -MatrixXd::Identity(3, 3);
  b.head(nr) = b_;
  b.tail<6>() = half_side_len * VectorXd::Ones(6);

  const int num_obs = 1;
  obs_vec.resize(num_obs);

  obs_vec[0] = std::make_shared<StaticPolytope<3>>(A, b, center, margin,
                                                   sc_modulus, true);
}

void GetRobotSafeRegion(std::shared_ptr<ConvexSet>& c1,
                        std::shared_ptr<ConvexSet>& c2, double max_vel) {
  // Quadrotor shape set
  const double pow = 2.5;
  Eigen::Vector4d coeff(1.0, 1.0, 0.4, 0.3);  // [m, m, m, ]
  double margin = 0.05;                       // [m]
  const std::shared_ptr<ConvexSet> shape =
      std::make_shared<QuadrotorShape>(pow, coeff, margin);

  // Quadrotor corridor set
  const double stop_time = 1.0;         // [s]
  const double orientation_cost = 0.5;  // [s]
  margin = 0.0;                         // [m]
  const std::shared_ptr<ConvexSet> corridor_1 =
      std::make_shared<QuadrotorCorridor>(stop_time, orientation_cost, max_vel,
                                          margin);
  const std::shared_ptr<ConvexSet> corridor_2 =
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

  // shape + corridor
  c1 = std::make_shared<MinkowskiSumSet>(shape, corridor_1);
  // downwash + corridor
  c2 = std::make_shared<MinkowskiSumSet>(downwash, corridor_2);
}

void SetupEnvironment(double dt, Environment& env) {
  // Set random seed
  std::srand(10);

  // Set obstacles (static polytopes)
  std::vector<std::shared_ptr<ConvexSet>> obs_vec;
  GetObstacle(obs_vec);
  VectorXd x = VectorXd::Zero(0);
  for (auto obs : obs_vec) obs->set_states(x, x);
  const int num_obs = static_cast<int>(obs_vec.size());
  // Set quadrotor safe regions
  const double max_vel = 1.0;  // [m/s]
  std::shared_ptr<ConvexSet> quad_c1, quad_c2;
  GetRobotSafeRegion(quad_c1, quad_c2, max_vel);

  // Set solver options and distance solver
  auto solver = std::make_shared<DistanceSolver>();
  MatrixXd metric = MatrixXd::Identity(3, 3);
  auto opt = std::make_shared<SolverOptions>();
  opt->metric = metric;
  opt->kkt_ode.use_kkt_err_tol = false;
  opt->kkt_ode.timestep = dt;

  // Set collision pairs
  std::vector<std::shared_ptr<CollisionPair>> cps;
  for (int i = 0; i < num_obs; ++i) {
    cps.push_back(
        std::make_shared<CollisionPair>(obs_vec[i], quad_c1, opt, solver));
    cps.push_back(
        std::make_shared<CollisionPair>(obs_vec[i], quad_c2, opt, solver));
  }

  // Set dynamical systems
  const double mass = 0.5;  // [kg].
  const Eigen::Vector3d inertia_diag(2.32 * 1e-3, 2.32 * 1e-3, 4 * 1e-3);
  const MatrixXd inertia = inertia_diag.asDiagonal();
  const MatrixXd constr_mat_u = MatrixXd::Zero(0, 4);
  const VectorXd constr_vec_u = VectorXd::Zero(0);

  std::shared_ptr<DynamicalSystem> sys =
      std::make_shared<Quadrotor>(mass, inertia, constr_mat_u, constr_vec_u);
  std::shared_ptr<DynamicalSystem> sys_r =
      std::make_shared<QuadrotorReduced>(mass, constr_mat_u, constr_vec_u);

  // Update environment struct
  env.num_obs = num_obs;
  env.cps = cps;
  env.sys = sys;
  env.sys_r = sys_r;
  env.mass = mass;
  env.inertia = inertia;
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

void GetDesiredTrajectory(double t, Environment& env, Trajectory& trajectory) {
  assert(t >= 0);

  const double height = 3.0;               // [m]
  const double init_x_pos = -7.5;          // [m]
  const double x_pos_lead = 0.5;           // [m]
  const double speed = 0.5;                // [m/s]
  const Eigen::Vector3d w(0.0, 0.0, 0.0);  // [rad/s]

  Eigen::Vector3d p, v, a;
  if (t <= 0.05) {
    p << init_x_pos, 0.0, height;
    v = Eigen::Vector3d::Zero();
  } else {
    const auto x = env.sys->x();
    p << x(0) + x_pos_lead, 0.0, height;
    v << speed, 0.0, 0.0;
  }
  a = Eigen::Vector3d::Zero();

  trajectory.time = t;
  trajectory.pd = p;
  trajectory.vd = v;
  trajectory.ad = a;
  trajectory.wd = w;
}

void GetReferenceControl(const Environment& env, const Trajectory& trajectory,
                         VectorXd& ur) {
  assert(ur.rows() == 4);

  const VectorXd x = env.sys_r->x();
  const auto p = x.head<3>();
  const auto v = x.segment<3>(3);
  const auto R = x.segment<9>(6).reshaped(3, 3);
  const auto pd = trajectory.pd;
  const auto vd = trajectory.vd;
  const auto ad = trajectory.ad;
  const auto wd = trajectory.wd;

  const double k_p = 0.5;
  const double k_d = 0.5;
  const double k_R = 3.0;
  const Eigen::Vector3d g(0.0, 0.0, 9.81);  // [m/s^2]
  const double max_acc = 3.0 / 4.0 * g.norm();

  VectorXd a_ref = ad - k_p * (p - pd) - k_d * (v - vd);
  const double norm = a_ref.norm();
  if (norm > max_acc) a_ref = a_ref / norm * max_acc;
  MatrixXd Rd(3, 3);
  RotationFromZVector(a_ref + g, Rd);
  // Thrust input
  ur(0) = env.mass * R.col(2).dot(a_ref + g);
  // Angular velocity input
  ur.tail<3>() = So3PTrackingControl(R, Rd, wd, {k_R, 0.0});
}

struct Logs {
  VectorXd t;  // (t_seq, t_0, T, dt)
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
  VectorXd margin2;
  int num_opt_solves;

  Logs(int Nlog, int num_cps, int nx)
      : t(Nlog),
        num_cps(num_cps),
        x(nx, Nlog),
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
        margin2(num_cps),
        num_opt_solves{0} {}

  void UpdateLogs(int i, const Environment& env, double solve_time_ode,
                  double solve_time_qp, const VectorXd& kkt_err);

  void SaveLogs(std::ofstream& outfile);
};

void Logs::UpdateLogs(int i, const Environment& env, double solve_time_ode,
                      double solve_time_qp, const VectorXd& kkt_err) {
  const int Nlog = static_cast<int>(t.rows());
  assert((i >= 0) && (i < Nlog));

  const int nz = env.cps[0]->get_set1()->nz() + env.cps[0]->get_set2()->nz();
  const int nr = env.cps[0]->get_set1()->nr() + env.cps[0]->get_set2()->nr();
  VectorXd z_ode(nz), z_opt(nz), lambda_ode(nr), lambda_opt(nr);

  x.col(i) = env.sys_r->x();
  this->solve_time_ode(i) = solve_time_ode;
  this->solve_time_qp(i) = solve_time_qp;

  auto start = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double solve_time = 0.0;
  for (int j = 0; j < num_cps; ++j) {
    // Store current KKT solution
    dist2_ode(j, i) = env.cps[j]->get_kkt_solution(z_ode, lambda_ode);
    Ddist2_ode(j, i) = env.cps[j]->GetMinimumDistanceDerivative();
    // Get ipopt solution
    start = std::chrono::high_resolution_clock::now();
    env.cps[j]->MinimumDistance();
    elapsed = std::chrono::high_resolution_clock::now() - start;
    solve_time +=
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
                .count()) *
        1e-6;
    dist2_opt(j, i) = env.cps[j]->get_kkt_solution(z_opt, lambda_opt);
    // Restore KKT solution
    env.cps[j]->set_kkt_solution(dist2_ode(j, i), z_ode, lambda_ode);

    // Error norms
    z_err_norm(j, i) = (z_ode - z_opt).norm();
    z_opt_norm(j, i) = z_opt.norm();
    lambda_err_norm(j, i) = (lambda_ode - lambda_opt).norm();
    lambda_opt_norm(j, i) = lambda_opt.norm();

    // num_opt_solves
    if (env.opt->kkt_ode.use_kkt_err_tol)
      num_opt_solves += (kkt_err(j) > env.opt->kkt_ode.max_inf_kkt_err);
    else
      num_opt_solves += (kkt_err(j) > env.opt->kkt_ode.max_primal_dual_gap);
  }
  solve_time_opt(i) = solve_time;
}

void Logs::SaveLogs(std::ofstream& outfile) {
  const int Nlog = static_cast<int>(t.rows());
  const int nx = static_cast<int>(x.rows());

  // Header
  outfile << "#collision pairs," << num_cps << std::endl
          << "#ipopt solves," << num_opt_solves << std::endl
          << "margin^2 (m^2)";
  for (int i = 0; i < num_cps; ++i) outfile << "," << margin2(i);
  outfile << std::endl << "t (s),";
  for (int i = 0; i < nx; ++i) outfile << "x_" << i << ",";
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
    for (int i = 0; i < nx; ++i) outfile << x(i, k) << ",";
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
  const double T = 50.0;   // [s]
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
  GetDesiredTrajectory(t_0, env, trajectory);
  env.set_state(trajectory.pd, trajectory.vd, Eigen::Matrix3d::Identity(),
                trajectory.wd);
  for (auto cp : env.cps) cp->MinimumDistance();
  const int num_cps = static_cast<int>(env.cps.size());

  // Initialize logs
  Logs log(Nlog, num_cps, 15);
  log.t = VectorXd::LinSpaced(Nlog, t_0, T);

  // Control loop
  CbfQpController controller(env);
  controller.get_margin2(log.margin2);

  auto start = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  VectorXd u_ref(4), u(4);
  VectorXd kkt_err(num_cps);
  double solve_time_ode = 0.0;
  double solve_time_qp = 0.0;
  printf("Entering control loop...\n");
  for (int i = 0, j = 0; i < N; ++i) {
    // if (i % control_freq == 0) {
    // Get desired trajectory
    GetDesiredTrajectory(t_seq(i), env, trajectory);
    // Get reference input
    GetReferenceControl(env, trajectory, u_ref);
    // Solve CBF-QP
    start = std::chrono::high_resolution_clock::now();
    controller.Control(env, u_ref, u);
    elapsed = std::chrono::high_resolution_clock::now() - start;
    solve_time_qp =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
                .count()) *
        1e-6;
    // }
    // Update KKT solutions
    VectorXd ur(4);
    ur(0) = u(0);
    ur.tail<3>() = env.sys->x().tail<3>();
    env.set_state(env.sys_r->x(), ur);
    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < num_cps; ++k) kkt_err(k) = env.cps[k]->KktStep();
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
    env.UpdateSystemState(u, dt);
  }
  printf("Control loop finished.\n");

  // Save logs to .csv file
  printf("Saving logs...\n");
  std::string filename = "cbf_data.csv";
  std::ofstream outfile(filename);
  log.SaveLogs(outfile);
  outfile.close();
  printf("Logs saved.\n");

  return EXIT_SUCCESS;
}
