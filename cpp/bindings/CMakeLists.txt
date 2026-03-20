pybind11_add_module(interpolatecpp_py
    module.cpp
    bind_tridiagonal.cpp
    bind_cubic_spline.cpp
    bind_smoothing_spline.cpp
    bind_acc_splines.cpp
    bind_smoothing_search.cpp
)

target_link_libraries(interpolatecpp_py
    PRIVATE
        interpolatecpp::interpolatecpp
)

install(TARGETS interpolatecpp_py
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
