pybind11_add_module(interpolatecpp_py
    module.cpp
    # Phase 1: Cubic Splines
    bind_tridiagonal.cpp
    bind_cubic_spline.cpp
    bind_smoothing_spline.cpp
    bind_acc_splines.cpp
    bind_smoothing_search.cpp
    # Phase 2: B-Splines
    bind_bspline.cpp
    # Phase 3: Motion Profiles
    bind_motion.cpp
    # Phase 4: Quaternion Interpolation
    bind_quaternion.cpp
    # Phase 5: Geometric Paths
    bind_paths.cpp
)

target_link_libraries(interpolatecpp_py
    PRIVATE
        interpolatecpp::interpolatecpp
)

install(TARGETS interpolatecpp_py
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
