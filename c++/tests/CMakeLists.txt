# Testing library (GoogleTest v1.15.2)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


## Add test executables
set(LIB_TEST_SOURCES
  "${PROJECT_SOURCE_DIR}/tests/lemke_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/utils/logsumexp_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/utils/matrixfunctions_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/utils/numericalgradient_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/geometry/geometry_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/collision/distance_test.cc"
)

add_executable(${LIB_CORE}-tests
  ${LIB_TEST_SOURCES}
)

target_link_libraries(${LIB_CORE}-tests
  PRIVATE
    ${LIB_CORE}
    GTest::gtest_main
)


include(GoogleTest)
gtest_discover_tests(${LIB_CORE}-tests)
