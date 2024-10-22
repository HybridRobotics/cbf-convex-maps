add_library(sccbf
    "${PROJECT_SOURCE_DIR}/src/lemke.cpp"
    "${PROJECT_SOURCE_DIR}/src/math_utils/numerical_derivatives.cpp")

target_include_directories(sccbf
    PUBLIC "${PROJECT_SOURCE_DIR}/include/"
    #[[PRIVATE <private_headers>]])

target_link_libraries(sccbf
    PUBLIC Eigen3::Eigen)

# install(TARGETS sccbf LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/lib/")

