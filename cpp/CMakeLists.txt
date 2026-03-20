cmake_minimum_required(VERSION 3.21)
project(InterpolateCpp
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "C++ port of InterpolatePy trajectory planning library"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(INTERPOLATECPP_BUILD_TESTS "Build tests" ON)
option(INTERPOLATECPP_BUILD_BINDINGS "Build pybind11 bindings" OFF)
option(INTERPOLATECPP_BUILD_EXAMPLES "Build examples" OFF)

# Version macros
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/interpolatecpp/version.hpp"
)

# Dependencies via FetchContent
include(FetchContent)

FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    SYSTEM
)

if(INTERPOLATECPP_BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1
        GIT_SHALLOW TRUE
    )
endif()

if(INTERPOLATECPP_BUILD_BINDINGS)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
        GIT_SHALLOW TRUE
    )
endif()

# Make Eigen available (suppress its tests and docs)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen)
set(BUILD_TESTING ON CACHE BOOL "" FORCE)

# Library sources
set(INTERPOLATECPP_SOURCES
    src/cubic_spline.cpp
    src/cubic_smoothing_spline.cpp
    src/cubic_spline_with_acc1.cpp
    src/cubic_spline_with_acc2.cpp
    src/smoothing_search.cpp
    # Phase 2: B-spline
    src/bspline.cpp
    src/cubic_bspline_interpolation.cpp
    src/bspline_interpolator.cpp
    src/approximation_bspline.cpp
    src/smoothing_cubic_bspline.cpp
    # Phase 3: Motion
    src/polynomial_trajectory.cpp
    src/double_s_trajectory.cpp
    src/trapezoidal_trajectory.cpp
    src/parabolic_blend_trajectory.cpp
    # Phase 4: Quaternion
    src/quaternion.cpp
    src/quaternion_spline.cpp
    src/squad_c2.cpp
    src/log_quaternion_interpolation.cpp
    # Phase 5: Path
    src/linear_path.cpp
    src/circular_path.cpp
    src/frenet_frame.cpp
    src/linear_traj.cpp
)

add_library(interpolatecpp ${INTERPOLATECPP_SOURCES})
add_library(interpolatecpp::interpolatecpp ALIAS interpolatecpp)

target_include_directories(interpolatecpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(interpolatecpp PUBLIC Eigen3::Eigen)

set_target_properties(interpolatecpp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_compile_options(interpolatecpp PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4
    >
)

# Tests
if(INTERPOLATECPP_BUILD_TESTS)
    enable_testing()
    FetchContent_MakeAvailable(Catch2)
    add_subdirectory(tests)
endif()

# Bindings
if(INTERPOLATECPP_BUILD_BINDINGS)
    FetchContent_MakeAvailable(pybind11)
    add_subdirectory(bindings)
endif()

# Examples
if(INTERPOLATECPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install rules
include(GNUInstallDirs)

install(TARGETS interpolatecpp
    EXPORT InterpolateCppTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/interpolatecpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/interpolatecpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT InterpolateCppTargets
    FILE InterpolateCppTargets.cmake
    NAMESPACE interpolatecpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/InterpolateCpp
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/InterpolateCppConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/InterpolateCppConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/InterpolateCpp
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/InterpolateCppConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/InterpolateCpp
)
