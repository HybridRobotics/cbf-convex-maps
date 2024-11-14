#include <benchmark/benchmark.h>

#include <Eigen/Core>
// #include <Eigen/Geometry>

#include "sccbf/collision/collision_pair.h"
#include "sccbf/collision/distance_solver.h"
#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/geometry/polytope.h"
#include "sccbf/solver_options.h"
#include "sccbf/utils/matrix_utils.h"

using namespace sccbf;

std::shared_ptr<ConvexSet> GetRandomPolytope(int nz, int nr) {
  assert((nz == 2) || (nz == 3));

  const VectorXd center = VectorXd::Zero(nz);
  MatrixXd A(nr, nz);
  VectorXd b(nr);
  const double in_radius = 0.1 * nz;
  RandomPolytope(center, in_radius, A, b);
  const double sc_modulus = 1e-2;
  if (nz == 2)
    return std::make_shared<Polytope<2>>(A, b, 0.0, sc_modulus, true);
  else
    return std::make_shared<Polytope<3>>(A, b, 0.0, sc_modulus, true);
}

inline void SetRandomState(std::shared_ptr<ConvexSet>& set,
                           const VectorXd& pos) {
  const int nz = set->nz();
  const int nx = set->nx();
  assert((nz == 2 && nx == 6) || (nz == 3 && nx == 12));
  assert(pos.rows() == nz);

  MatrixXd rotation(nz, nz);
  RandomRotation(rotation);
  VectorXd x_new(set->nx());
  x_new.head(nz) = pos;
  x_new.tail(nz * nz) = rotation.reshaped(nz * nz, 1);
  set->set_x(x_new);
}

static void BM_Distance(benchmark::State& state) {
  const int nz = state.range(0);
  const int nr1 = 10;
  const int nr2 = 15;

  std::shared_ptr<DistanceSolver> solver = std::make_shared<DistanceSolver>();
  MatrixXd metric(nz, nz);
  const double eps = 1.0;
  RandomSpdMatrix(metric, eps);
  std::shared_ptr<SolverOptions> opt = std::make_shared<SolverOptions>();
  opt->metric = metric;

  auto C1 = GetRandomPolytope(nz, nr1);
  auto C2 = GetRandomPolytope(nz, nr2);
  VectorXd translation = 10.0 * VectorXd::Ones(nz) + VectorXd::Random(nz);
  SetRandomState(C1, translation);
  SetRandomState(C2, VectorXd::Zero(nz));

  auto cp = CollisionPair(C1, C2, opt, solver);

  VectorXd z(2 * nz), lambda(nr1 + nr2);
  double dist2{0};

  for (auto _ : state) {
    benchmark::DoNotOptimize(cp.MinimumDistance());
  }
}

BENCHMARK(BM_Distance)
    ->Unit(benchmark::kMicrosecond)
    // ->Repetitions(10)
    // ->ReportAggregatesOnly(true)
    ->MinWarmUpTime(1e-2)  // Warmup time (in sec) for ipopt.
    ->DenseRange(2, 3);    // nz = {2, 3}.

BENCHMARK_MAIN();
