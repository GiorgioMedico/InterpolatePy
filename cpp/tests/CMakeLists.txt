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
