# Benchmark library (Google Benchmark (living at head))
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG main # need main for benchmark::benchmark
)

# Do not build benchmark tests
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

FetchContent_MakeAvailable(googlebenchmark)


macro(add_benchmark_executable src)
  add_executable(${src}
    "${PROJECT_SOURCE_DIR}/benchmarks/${src}.cc"
  )
  target_link_libraries(${src}
    PRIVATE
      ${LIB_CORE}
      benchmark::benchmark
  )
endmacro()

add_benchmark_executable(dense_ldlt_bm)
add_benchmark_executable(lemke_bm)
add_benchmark_executable(distance_bm)
add_benchmark_executable(kktode_bm)
