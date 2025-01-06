#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sccbf/collision/collision_pair.h"
#include "sccbf/collision/distance_solver.h"
#include "sccbf/data_types.h"
#include "sccbf/geometry/convex_set.h"
#include "sccbf/geometry/ellipsoid.h"
#include "sccbf/geometry/polytope.h"
#include "sccbf/solver_options.h"
#include "sccbf/utils/matrix_utils.h"

using namespace sccbf;

std::shared_ptr<ConvexSet> GetRandomPolytope(int nz, int nr,
                                             double sc_modulus) {
  assert((nz == 2) || (nz == 3));

  const VectorXd center = VectorXd::Zero(nz);
  MatrixXd A(nr, nz);
  VectorXd b(nr);
  const double in_radius = 1.0;
  RandomPolytope(center, in_radius, A, b);
  if (nz == 2)
    return std::make_shared<Polytope<2>>(A, b, 0.0, sc_modulus, true);
  else
    return std::make_shared<Polytope<3>>(A, b, 0.0, sc_modulus, true);
}

std::shared_ptr<ConvexSet> GetRandomEllipsoid(int nz) {
  assert((nz == 2) || (nz == 3));

  MatrixXd Q(nz, nz);
  const double eps = 5e-1;
  RandomSpdMatrix(Q, eps);
  if (nz == 2)
    return std::make_shared<Ellipsoid<2>>(Q, 0.0);
  else
    return std::make_shared<Ellipsoid<3>>(Q, 0.0);
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
  const VectorXd dx_new = VectorXd::Random(set->ndx());
  set->set_states(x_new, dx_new);
}

static void BM_KktOde(benchmark::State& state) {
  const int nz = state.range(0);
  const int nr1 = 10;
  const int nr2 = 15;

  std::shared_ptr<DistanceSolver> solver = std::make_shared<DistanceSolver>();
  MatrixXd metric(nz, nz);
  const double eps = 1.0;
  RandomSpdMatrix(metric, eps);
  std::shared_ptr<SolverOptions> opt = std::make_shared<SolverOptions>();
  opt->metric = metric;
  opt->kkt_ode.use_kkt_err_tol = state.range(1);

  auto C1 = GetRandomPolytope(nz, nr1, 1e-2);
  auto C2 = GetRandomPolytope(nz, nr2, 0);
  if (state.range(2) == 0) C2 = GetRandomEllipsoid(nz);
  VectorXd translation = 10.0 * VectorXd::Ones(nz) + VectorXd::Random(nz);
  SetRandomState(C1, translation);
  SetRandomState(C2, VectorXd::Zero(nz));

  auto cp = std::make_shared<CollisionPair>(C1, C2, opt, solver);

  cp->MinimumDistance();

  for (auto _ : state) {
    benchmark::DoNotOptimize(cp->KktStep());
  }
}

BENCHMARK(BM_KktOde)
    ->Unit(benchmark::kMicrosecond)
    // ->Repetitions(10)
    // ->ReportAggregatesOnly(true)
    ->ArgsProduct({
        {2, 3},         // nz = {2, 3}.
        {true, false},  // use_kkt_err_tol = {true, false}.
        {0, 1}          // C2 = Ellipsoid (if 0), else Polytope.
    });

BENCHMARK_MAIN();
