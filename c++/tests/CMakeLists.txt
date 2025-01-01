# Testing library (GoogleTest v1.15.2)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


set(LIB_TEST_SOURCES_COLLISION
  "${PROJECT_SOURCE_DIR}/tests/collision/distance_test.cc"
)

set(LIB_TEST_SOURCES_GEOMETRY
  "${PROJECT_SOURCE_DIR}/tests/geometry/geometry_test.cc"
)

set(LIB_TEST_SOURCES_SYSTEM
  "${PROJECT_SOURCE_DIR}/tests/system/system_test.cc"
)

set(LIB_TEST_SOURCES_UTILS
  "${PROJECT_SOURCE_DIR}/tests/utils/logsumexp_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/utils/matrixfunctions_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/utils/numericalgradient_test.cc"
)

set(LIB_TEST_SOURCES_OTHERS
  "${PROJECT_SOURCE_DIR}/tests/lemke_test.cc"
)

add_executable(${LIB_CORE}_tests
  ${LIB_TEST_SOURCES_COLLISION}
  ${LIB_TEST_SOURCES_GEOMETRY}
  ${LIB_TEST_SOURCES_SYSTEM}
  ${LIB_TEST_SOURCES_UTILS}
  ${LIB_TEST_SOURCES_OTHERS}
)

target_link_libraries(${LIB_CORE}_tests
  PRIVATE
    ${LIB_CORE}
    GTest::gtest_main
)


include(GoogleTest)
gtest_discover_tests(${LIB_CORE}_tests)


## Standalone tests.
add_executable(${LIB_CORE}_so3_tracking_test
  "${PROJECT_SOURCE_DIR}/tests/utils/so3_tracking_test.cc"
)

target_link_libraries(${LIB_CORE}_so3_tracking_test
  PRIVATE
    ${LIB_CORE}
)
