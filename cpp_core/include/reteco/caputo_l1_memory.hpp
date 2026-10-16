#pragma once
#include "reteco/time_grid.hpp"
#include <Eigen/Dense>
#include <vector>

namespace reteco {

class CaputoL1Memory {
public:
    CaputoL1Memory(double gamma_order, const TimeGrid& grid, std::size_t num_dofs);

    void reset(const Eigen::VectorXd& u0);

    double current_implicit_coeff(std::size_t n) const;

    std::vector<double> step_weights(std::size_t n) const;

    Eigen::VectorXd compute_history_load(std::size_t n) const;

    Eigen::VectorXd evaluate_discrete_derivative(std::size_t n, const Eigen::VectorXd& u_n) const;

    void commit_step(std::size_t n, const Eigen::VectorXd& u_n);

    double order() const { return gamma_; }
    std::size_t num_dofs() const { return num_dofs_; }

private:
    double gamma_;
    double inv_gamma_2_minus_g_;
    TimeGrid grid_;
    std::size_t num_dofs_;
    std::size_t committed_step_ = 0;
    bool is_initialized_ = false;

    Eigen::VectorXd u_prev_;
    Eigen::MatrixXd increments_;

    void validate_step_index(std::size_t n) const;
};

} // namespace reteco