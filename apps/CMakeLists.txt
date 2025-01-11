## Example 1: KKT ODE
add_executable(${LIB_CORE}_kkt_ode
  "${PROJECT_SOURCE_DIR}/apps/kkt_ode_example/kkt_ode.cc"
)

target_link_libraries(${LIB_CORE}_kkt_ode
  PRIVATE
    ${LIB_CORE}
)

## Example 2: Multi-agent Strongly convex map CBF
find_package(OsqpEigen)

add_executable(${LIB_CORE}_ma_cbf
  "${PROJECT_SOURCE_DIR}/apps/ma_cbf_example/ma_cbf_example.cc"
)

target_include_directories(${LIB_CORE}_ma_cbf
  PRIVATE
    "${PROJECT_SOURCE_DIR}/apps/"
)

target_link_libraries(${LIB_CORE}_ma_cbf
  PRIVATE
    ${LIB_CORE}
    OsqpEigen::OsqpEigen
)
