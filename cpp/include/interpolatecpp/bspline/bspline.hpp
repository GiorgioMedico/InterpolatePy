#pragma once

#include <Eigen/Core>
#include <span>

#include <interpolatecpp/config.hpp>

namespace interpolatecpp::bspline {

/// B-spline curve of arbitrary degree supporting N-dimensional control points.
///
/// Implements the Cox-de Boor algorithm for evaluation and derivative computation.
/// Control points are stored as rows of a matrix (n_points x dimension).
class INTERPOLATECPP_API BSpline {
  public:
    /// Construct a B-spline curve.
    ///
    /// @param degree         Degree of the B-spline (must be non-negative)
    /// @param knots          Non-decreasing knot vector
    /// @param control_points Control points (n_points x dimension matrix, or vector for 1D)
    /// @throws std::invalid_argument if inputs violate B-spline requirements
    BSpline(int degree, std::span<const double> knots, const Eigen::MatrixXd& control_points);

    virtual ~BSpline() = default;

    // Evaluation (satisfies CurveEvaluator concept)
    [[nodiscard]] Eigen::VectorXd evaluate(double u) const;
    [[nodiscard]] Eigen::VectorXd evaluate_derivative(double u, int order = 1) const;

    // Curve generation
    [[nodiscard]] std::pair<Eigen::VectorXd, Eigen::MatrixXd>
    generate_curve_points(int num_points = 100) const;

    // Knot span and basis functions
    [[nodiscard]] int find_knot_span(double u) const;
    [[nodiscard]] Eigen::VectorXd basis_functions(double u, int span_index) const;
    [[nodiscard]] Eigen::MatrixXd basis_function_derivatives(double u, int span_index,
                                                             int order) const;

    // Static knot vector generators
    [[nodiscard]] static Eigen::VectorXd create_uniform_knots(int degree,
                                                              int num_control_points,
                                                              double domain_min = 0.0,
                                                              double domain_max = 1.0);

    [[nodiscard]] static Eigen::VectorXd create_periodic_knots(int degree,
                                                               int num_control_points,
                                                               double domain_min = 0.0,
                                                               double domain_max = 1.0);

    // Accessors
    [[nodiscard]] int degree() const noexcept { return degree_; }
    [[nodiscard]] const Eigen::VectorXd& knots() const noexcept { return knots_; }
    [[nodiscard]] const Eigen::MatrixXd& control_points() const noexcept { return control_points_; }
    [[nodiscard]] double u_min() const noexcept { return u_min_; }
    [[nodiscard]] double u_max() const noexcept { return u_max_; }
    [[nodiscard]] int dimension() const noexcept { return dimension_; }
    [[nodiscard]] int n_control_points() const noexcept {
        return static_cast<int>(control_points_.rows());
    }

  protected:
    /// Protected constructor for derived classes that will initialize fields themselves.
    struct DeferInit {};
    explicit BSpline(DeferInit) : degree_(0), u_min_(0), u_max_(0), dimension_(0) {}

    int degree_;
    Eigen::VectorXd knots_;
    Eigen::MatrixXd control_points_;  // (n_points x dimension)
    double u_min_;
    double u_max_;
    int dimension_;

    static constexpr double kEps = 1e-10;
};

}  // namespace interpolatecpp::bspline
