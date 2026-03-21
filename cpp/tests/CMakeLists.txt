include(Catch2Testing OPTIONAL RESULT_VARIABLE CATCH2_TESTING_MODULE)
if(NOT CATCH2_TESTING_MODULE)
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
endif()
include(CTest)
include(Catch)

set(TEST_SOURCES
    test_tridiagonal.cpp
    test_cubic_spline.cpp
    test_cubic_smoothing_spline.cpp
    test_cubic_spline_with_acc1.cpp
    test_cubic_spline_with_acc2.cpp
    test_smoothing_search.cpp
    test_concepts.cpp
    # Phase 2: B-spline
    test_bspline.cpp
    test_bspline_variants.cpp
    # Phase 3: Motion
    test_polynomial_trajectory.cpp
    test_double_s_trajectory.cpp
    test_trapezoidal_trajectory.cpp
    test_parabolic_blend_trajectory.cpp
    # Phase 4: Quaternion
    test_quaternion.cpp
    test_quaternion_spline.cpp
    # Phase 5: Path
    test_paths.cpp
)

add_executable(interpolatecpp_tests ${TEST_SOURCES})

target_link_libraries(interpolatecpp_tests
    PRIVATE
        interpolatecpp::interpolatecpp
        Catch2::Catch2WithMain
)

target_include_directories(interpolatecpp_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shared
)

target_compile_options(interpolatecpp_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wpedantic
    >
)

catch_discover_tests(interpolatecpp_tests)
