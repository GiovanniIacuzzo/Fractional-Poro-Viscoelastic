#include "reteco/caputo_l1_memory.hpp"
#include <cmath>
#include <stdexcept>

namespace reteco {

CaputoL1Memory::CaputoL1Memory(double gamma_order, const TimeGrid& grid, std::size_t num_dofs)
    : gamma_(gamma_order),
      inv_gamma_2_minus_g_(1.0 / std::tgamma(2.0 - gamma_order)),
      grid_(grid),
      num_dofs_(num_dofs),
      committed_step_(0),
      is_initialized_(false),
      u_prev_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(num_dofs))),
      increments_(Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(num_dofs),
          static_cast<Eigen::Index>(grid.num_steps()))) {
    if (!std::isfinite(gamma_) || !(gamma_ > 0.0 && gamma_ < 1.0)) {
        throw std::invalid_argument("CaputoL1Memory: gamma_order must be finite and in (0, 1).");
    }
    if (num_dofs_ == 0) {
        throw std::invalid_argument("CaputoL1Memory: num_dofs must be >= 1.");
    }
}

void CaputoL1Memory::validate_step_index(std::size_t n) const {
    if (!is_initialized_) {
        throw std::runtime_error("CaputoL1Memory: memory not initialized. Call reset(u0) first.");
    }
    if (n < 1 || n > grid_.num_steps()) {
        throw std::out_of_range("CaputoL1Memory: step index n out of range [1, num_steps].");
    }
}

void CaputoL1Memory::reset(const Eigen::VectorXd& u0) {
    if (static_cast<std::size_t>(u0.size()) != num_dofs_) {
        throw std::invalid_argument("CaputoL1Memory::reset: u0 size mismatch.");
    }
    if (!u0.allFinite()) {
        throw std::invalid_argument("CaputoL1Memory::reset: u0 contains non-finite values.");
    }
    u_prev_ = u0;
    increments_.setZero();
    committed_step_ = 0;
    is_initialized_ = true;
}

double CaputoL1Memory::current_implicit_coeff(std::size_t n) const {
    if (n < 1 || n > grid_.num_steps()) {
        throw std::out_of_range("CaputoL1Memory::current_implicit_coeff: step index n out of range.");
    }
    const double dt_n = grid_.dt(n);
    return inv_gamma_2_minus_g_ * std::pow(dt_n, -gamma_);
}

std::vector<double> CaputoL1Memory::step_weights(std::size_t n) const {
    if (n < 1 || n > grid_.num_steps()) {
        throw std::out_of_range("CaputoL1Memory::step_weights: invalid step index n.");
    }
    std::vector<double> w(n, 0.0);
    const double t_n = grid_.time_at(n);
    const double one_minus_g = 1.0 - gamma_;

    for (std::size_t k = 1; k <= n; ++k) {
        const double t_km1 = grid_.time_at(k - 1);
        const double t_k   = grid_.time_at(k);
        const double dt_k  = t_k - t_km1;
        const double diff  = std::pow(t_n - t_km1, one_minus_g) - std::pow(t_n - t_k, one_minus_g);
        w[k - 1] = inv_gamma_2_minus_g_ * (diff / dt_k);
    }
    return w;
}

Eigen::VectorXd CaputoL1Memory::compute_history_load(std::size_t n) const {
    validate_step_index(n);
    if (n != committed_step_ + 1) {
        throw std::runtime_error("CaputoL1Memory::compute_history_load: out-of-order time step.");
    }
    const double d_nn = current_implicit_coeff(n);
    if (n == 1) {
        return d_nn * u_prev_;
    }

    const std::vector<double> w = step_weights(n);
    Eigen::VectorXd w_hist(static_cast<Eigen::Index>(n - 1));
    for (std::size_t k = 1; k < n; ++k) {
        w_hist(static_cast<Eigen::Index>(k - 1)) = w[k - 1];
    }

    const Eigen::VectorXd sum_past =
        increments_.leftCols(static_cast<Eigen::Index>(n - 1)) * w_hist;
    return d_nn * u_prev_ - sum_past;
}

Eigen::VectorXd CaputoL1Memory::evaluate_discrete_derivative(
    std::size_t n, const Eigen::VectorXd& u_n) const {
    validate_step_index(n);
    if (static_cast<std::size_t>(u_n.size()) != num_dofs_) {
        throw std::invalid_argument("CaputoL1Memory::evaluate_discrete_derivative: u_n size mismatch.");
    }
    const double d_nn = current_implicit_coeff(n);
    const Eigen::VectorXd H_nm1 = compute_history_load(n);
    return d_nn * u_n - H_nm1;
}

void CaputoL1Memory::commit_step(std::size_t n, const Eigen::VectorXd& u_n) {
    validate_step_index(n);
    if (n != committed_step_ + 1) {
        throw std::runtime_error("CaputoL1Memory::commit_step: out-of-order time step.");
    }
    if (static_cast<std::size_t>(u_n.size()) != num_dofs_) {
        throw std::invalid_argument("CaputoL1Memory::commit_step: u_n size mismatch.");
    }
    if (!u_n.allFinite()) {
        throw std::invalid_argument("CaputoL1Memory::commit_step: u_n contains non-finite values.");
    }
    increments_.col(static_cast<Eigen::Index>(n - 1)) = u_n - u_prev_;
    u_prev_ = u_n;
    committed_step_ = n;
}

} // namespace reteco