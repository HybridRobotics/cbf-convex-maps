## Add library for sources in the apps/.
set(LIB_APPS sccbf_apps)

add_library(${LIB_APPS}
  "${PROJECT_SOURCE_DIR}/apps/src/quadrotor_corridor.cc"
  "${PROJECT_SOURCE_DIR}/apps/src/quadrotor_downwash.cc"
  "${PROJECT_SOURCE_DIR}/apps/src/quadrotor_shape.cc"
  "${PROJECT_SOURCE_DIR}/apps/src/quadrotor_uncertainty.cc"
)

target_include_directories(${LIB_APPS}
  PUBLIC
    "${PROJECT_SOURCE_DIR}/apps/include/"
    "${IPOPT_INCLUDE_DIRS}"
)

target_link_libraries(${LIB_APPS}
  PUBLIC
    Eigen3::Eigen
    ${LIB_CORE}
)


## Add tests.
if(BUILD_TESTS)
  add_executable(${LIB_APPS}_tests
    "${PROJECT_SOURCE_DIR}/apps/tests/geometry_test.cc"
  )

  target_link_libraries(${LIB_APPS}_tests
  PRIVATE
    ${LIB_APPS}
    GTest::gtest_main
  )

  include(GoogleTest)
  gtest_discover_tests(${LIB_APPS}_tests)
endif()


## Example 1: KKT ODE
add_executable(${LIB_APPS}_kkt_ode
  "${PROJECT_SOURCE_DIR}/apps/kkt_ode_example/kkt_ode.cc"
)

target_link_libraries(${LIB_APPS}_kkt_ode
  PRIVATE
    ${LIB_APPS}
)
