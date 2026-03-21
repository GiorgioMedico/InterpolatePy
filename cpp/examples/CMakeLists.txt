set(EXAMPLE_SOURCES
    # Phase 1: Cubic Splines
    cubic_spline_example.cpp
    cubic_spline_acc1_example.cpp
    cubic_spline_acc2_example.cpp
    cubic_smoothing_example.cpp
    # Phase 2: B-Splines
    bspline_example.cpp
    bspline_cubic_example.cpp
    bspline_interpolator_example.cpp
    bspline_approx_smooth_example.cpp
    # Phase 3: Motion Profiles
    trapezoidal_example.cpp
    polynomial_example.cpp
    double_s_example.cpp
    parabolic_linear_example.cpp
    # Phase 4: Quaternion
    quaternion_example.cpp
    # Phase 5: Paths & Concepts
    paths_example.cpp
    concepts_example.cpp
)

foreach(source IN LISTS EXAMPLE_SOURCES)
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE interpolatecpp::interpolatecpp)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shared)
endforeach()
