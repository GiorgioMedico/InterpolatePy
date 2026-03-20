#include <interpolatecpp/spline/cubic_spline.hpp>

#include <iostream>
#include <vector>

int main() {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> q = {0.0, 1.0, 0.0, 1.0};

    interpolatecpp::spline::CubicSpline spline(t, q);

    std::cout << "CubicSpline evaluation:\n";
    for (double ti = 0.0; ti <= 3.0; ti += 0.5) {
        std::cout << "  t=" << ti << "  pos=" << spline.evaluate(ti)
                  << "  vel=" << spline.evaluate_velocity(ti)
                  << "  acc=" << spline.evaluate_acceleration(ti) << "\n";
    }

    return 0;
}
