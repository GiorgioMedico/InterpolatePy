#pragma once

#include <Eigen/Core>

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace interpolatecpp::examples {

inline void print_header(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n\n";
}

inline void print_separator(char ch = '-', int width = 60) {
    std::cout << std::string(static_cast<size_t>(width), ch) << "\n";
}

inline void print_value(const std::string& label, double value, int precision = 6) {
    std::cout << "  " << label << ": " << std::fixed << std::setprecision(precision) << value
              << "\n";
}

inline void print_vector(const std::string& label, const Eigen::VectorXd& v, int precision = 4) {
    std::cout << "  " << label << ": [";
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(precision) << v(i);
    }
    std::cout << "]\n";
}

inline void print_vector3(const std::string& label, const Eigen::Vector3d& v,
                           int precision = 4) {
    std::cout << "  " << label << ": (" << std::fixed << std::setprecision(precision) << v.x()
              << ", " << v.y() << ", " << v.z() << ")\n";
}

/// Print a trajectory table with columns: Time | Position | Velocity | Acceleration.
inline void print_trajectory_table(
    const std::function<double(double)>& pos_fn,
    const std::function<double(double)>& vel_fn,
    const std::function<double(double)>& acc_fn, double t_start, double t_end,
    int num_samples = 15) {
    const int w = 14;
    const int p = 6;

    std::cout << std::right << std::setw(w) << "Time" << std::setw(w) << "Position"
              << std::setw(w) << "Velocity" << std::setw(w) << "Acceleration"
              << "\n";
    print_separator('-', 4 * w);

    for (int i = 0; i <= num_samples; ++i) {
        double t = t_start + (t_end - t_start) * static_cast<double>(i) / num_samples;
        std::cout << std::fixed << std::setprecision(p) << std::setw(w) << t << std::setw(w)
                  << pos_fn(t) << std::setw(w) << vel_fn(t) << std::setw(w) << acc_fn(t)
                  << "\n";
    }
    std::cout << "\n";
}

/// Print trajectory table with jerk (4 columns + jerk).
inline void print_full_trajectory_table(
    const std::function<std::tuple<double, double, double, double>(double)>& eval_fn,
    double t_start, double t_end, int num_samples = 15) {
    const int w = 14;
    const int p = 6;

    std::cout << std::right << std::setw(w) << "Time" << std::setw(w) << "Position"
              << std::setw(w) << "Velocity" << std::setw(w) << "Acceleration" << std::setw(w)
              << "Jerk"
              << "\n";
    print_separator('-', 5 * w);

    for (int i = 0; i <= num_samples; ++i) {
        double t = t_start + (t_end - t_start) * static_cast<double>(i) / num_samples;
        auto [pos, vel, acc, jrk] = eval_fn(t);
        std::cout << std::fixed << std::setprecision(p) << std::setw(w) << t << std::setw(w)
                  << pos << std::setw(w) << vel << std::setw(w) << acc << std::setw(w) << jrk
                  << "\n";
    }
    std::cout << "\n";
}

/// Print a matrix row by row.
inline void print_matrix(const std::string& label, const Eigen::MatrixXd& m, int precision = 4) {
    std::cout << "  " << label << " (" << m.rows() << "x" << m.cols() << "):\n";
    for (Eigen::Index i = 0; i < m.rows(); ++i) {
        std::cout << "    [";
        for (Eigen::Index j = 0; j < m.cols(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(precision) << std::setw(10) << m(i, j);
        }
        std::cout << "]\n";
    }
}

}  // namespace interpolatecpp::examples
