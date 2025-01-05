set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(IPOPT 3.11.9 REQUIRED)
# Minimum required version of Ipopt (from Ifopt).
# previous IPOPT versions have a bug which causes mumps segfault:
# https://bugs.launchpad.net/ubuntu/+source/coinor-ipopt/+bug/1167585


set(LIB_CORE_SOURCES_COLLISION
  "${PROJECT_SOURCE_DIR}/src/collision/distance_solver.cc"
  "${PROJECT_SOURCE_DIR}/src/collision/collision_pair.cc"
)

set(LIB_CORE_SOURCES_GEOMETRY
  "${PROJECT_SOURCE_DIR}/src/geometry/ellipsoid.cc"
  "${PROJECT_SOURCE_DIR}/src/geometry/polytope.cc"
  "${PROJECT_SOURCE_DIR}/src/geometry/quadrotor_corridor.cc"
  "${PROJECT_SOURCE_DIR}/src/geometry/quadrotor_downwash.cc"
  "${PROJECT_SOURCE_DIR}/src/geometry/quadrotor_shape.cc"
  "${PROJECT_SOURCE_DIR}/src/geometry/quadrotor_uncertainty.cc"
  "${PROJECT_SOURCE_DIR}/src/geometry/static_ellipsoid.cc"
  "${PROJECT_SOURCE_DIR}/src/geometry/static_polytope.cc"
)

set(LIB_CORE_SOURCES_SYSTEM
  "${PROJECT_SOURCE_DIR}/src/system/quadrotor.cc"
  "${PROJECT_SOURCE_DIR}/src/system/quadrotor_reduced.cc"
)

set(LIB_CORE_SOURCES_TRANSFORMATION
  "${PROJECT_SOURCE_DIR}/src/transformation/intersection.cc"
  "${PROJECT_SOURCE_DIR}/src/transformation/minkowski.cc"
)

set(LIB_CORE_SOURCES_UTILS
  "${PROJECT_SOURCE_DIR}/src/utils/numerical_derivatives.cc"
)

set(LIB_CORE_SOURCES_OTHERS
  "${PROJECT_SOURCE_DIR}/src/lemke.cc"
)

add_library(${LIB_CORE}
  ${LIB_CORE_SOURCES_COLLISION}
  ${LIB_CORE_SOURCES_GEOMETRY}
  ${LIB_CORE_SOURCES_SYSTEM}
  ${LIB_CORE_SOURCES_TRANSFORMATION}
  ${LIB_CORE_SOURCES_UTILS}
  ${LIB_CORE_SOURCES_OTHERS}
)

target_include_directories(${LIB_CORE}
  PUBLIC
    "${PROJECT_SOURCE_DIR}/include/"
  # PRIVATE
    ${IPOPT_INCLUDE_DIRS}
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

# install(TARGETS ${LIB_CORE} LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/lib/")

