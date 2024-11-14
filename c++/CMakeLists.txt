## CMake version
cmake_minimum_required(VERSION 3.15)

## User flags
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS
  "Build benchmarks (only built if BUILD_TESTS is enabled)"
  ON
)
option(BUILD_APPS "Build examples" ON)
option(BUILD_DOCS "Add documentation" OFF)

## Project name and version
project(
  StronglyConvexCBF
  VERSION 0.1.0
  DESCRIPTION "CBFs for strongly convex maps"
  LANGUAGES CXX
)

## Compiler flags
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} \
  -pedantic"
)
set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} \
  -O0 -Wall -Wextra -Wconversion -g"
)
# -ffast-math (-fno-math-errno, -Ofast) not used because it breaks IEEE compliance
# TODO: Test -mavx2
set(CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE} \
  -O3 -Werror -DNDEBUG -funroll-loops -march=native -mfpmath=sse -fopenmp"
)

## Packages
include(FetchContent)

find_package(Eigen3 3.4.90 REQUIRED NO_MODULE)


## SCCBF subdirectories
set(LIB_CORE sccbf)

add_subdirectory(src)

if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
  if (BUILD_BENCHMARKS AND CMAKE_BUILD_TYPE STREQUAL "Release")
    add_subdirectory(benchmarks)
  endif()
endif()

if (BUILD_APPS)
  add_subdirectory(apps)
endif()

if (BUILD_DOCS)
  find_package(Doxygen)
  if (Doxygen_FOUND)
    # TODO: add_subdirectory(docs)
  else()
    message(STATUS "Doxygen not found; not building docs")
  endif()
endif()

