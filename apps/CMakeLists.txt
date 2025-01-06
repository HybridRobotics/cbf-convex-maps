## Example 1: KKT ODE
add_executable(${LIB_CORE}_kkt_ode
  "${PROJECT_SOURCE_DIR}/apps/kkt_ode_example/kkt_ode.cc"
)

target_link_libraries(${LIB_CORE}_kkt_ode
  PRIVATE
    ${LIB_CORE}
)

## Example 2: Strongly convex map CBF
find_package(OsqpEigen)

add_executable(${LIB_CORE}_cbf
  "${PROJECT_SOURCE_DIR}/apps/cbf_example/quadrotor_cbf_example.cc"
)

target_include_directories(${LIB_CORE}_cbf
  PRIVATE
    "${PROJECT_SOURCE_DIR}/apps/"
)

target_link_libraries(${LIB_CORE}_cbf
  PRIVATE
    ${LIB_CORE}
    OsqpEigen::OsqpEigen
)
