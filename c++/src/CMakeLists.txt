set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(IPOPT 3.11.9 REQUIRED)
# Minimum required version of Ipopt (from Ifopt).
# previous IPOPT versions have a bug which causes mumps segfault:
# https://bugs.launchpad.net/ubuntu/+source/coinor-ipopt/+bug/1167585


add_library(${LIB_CORE}
  "${PROJECT_SOURCE_DIR}/src/lemke.cc"
  "${PROJECT_SOURCE_DIR}/src/math_utils/numerical_derivatives.cc"
  "${PROJECT_SOURCE_DIR}/src/collision/distance_solver.cc"
  "${PROJECT_SOURCE_DIR}/src/collision/collision_pair.cc"
)

target_include_directories(${LIB_CORE}
  PUBLIC
    "${PROJECT_SOURCE_DIR}/include/"
  PRIVATE
    "${IPOPT_INCLUDE_DIRS}"
)

target_link_libraries(${LIB_CORE}
  PRIVATE
    Eigen3::Eigen
    ${IPOPT_LIBRARIES}
    ${IPOPT_LINK_FLAGS}
)

target_compile_definitions(${LIB_CORE}
  PRIVATE
    ${IPOPT_DEFINITIONS}
)

# install(TARGETS sccbf LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/lib/")

