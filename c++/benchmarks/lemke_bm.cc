#include <benchmark/benchmark.h>

#include "sccbf/data_types.h"
#include "sccbf/lemke.h"
#include "sccbf/solver_options.h"
#include "sccbf/utils/matrix_utils.h"

using namespace sccbf;

static void BM_Lemke(benchmark::State& state) {
  const int n = state.range(0);
  const double eps = 1e-7;

  MatrixXd M(n, n);
  RandomSpdMatrix(M, eps);
  VectorXd q(n), z(n);
  q.array() = VectorXd::Random(n).array() - 1.0;

  for (auto _ : state) {
    benchmark::DoNotOptimize(SolveLcp(M, q, z, LcpOptions{}));
  }
}

BENCHMARK(BM_Lemke)
    ->Unit(benchmark::kMicrosecond)
    // ->Repetitions(10)
    // ->ReportAggregatesOnly(true)
    ->RangeMultiplier(2)
    ->Range(2, 8);  // n = {2, 4, 8}.

BENCHMARK_MAIN();
