#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sccbf/data_types.h"

using namespace sccbf;

static void BM_DenseLDLT(benchmark::State& state) {
  const int n = state.range(0);
  const int m = 1;
  const double eps = 1e-4;

  MatrixXd A(n, n);
  MatrixXd mat_ = MatrixXd::Random(n, n);
  A = eps * MatrixXd::Identity(n, n) + mat_ * mat_.transpose();
  MatrixXd b(n, m), x(n, m);
  b = MatrixXd::Random(n, m);
  Eigen::LDLT<MatrixXd> ldlt(n);

  for (auto _ : state) {
    ldlt.compute(A);
    benchmark::DoNotOptimize(x = ldlt.solve(b));
    A = A + eps * MatrixXd::Identity(n, n);
  }
}

BENCHMARK(BM_DenseLDLT)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(5, 40);

BENCHMARK_MAIN();
