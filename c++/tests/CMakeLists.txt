# Testing library (GoogleTest v1.15.2)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


## Add test executables
add_executable(lemke_test "${PROJECT_SOURCE_DIR}/tests/lemke_unittest.cpp")
add_executable(math_utils_test
  "${PROJECT_SOURCE_DIR}/tests/math_utils/logsumexp_test.cpp"
  "${PROJECT_SOURCE_DIR}/tests/math_utils/matrixfunctions_test.cpp"
  "${PROJECT_SOURCE_DIR}/tests/math_utils/numericalgradient_test.cpp")


## Link libraries
target_link_libraries(lemke_test PRIVATE sccbf GTest::gtest_main)
target_link_libraries(math_utils_test PRIVATE sccbf GTest::gtest_main)


include(GoogleTest)
gtest_discover_tests(lemke_test)
gtest_discover_tests(math_utils_test)

