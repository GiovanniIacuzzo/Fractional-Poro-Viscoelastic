#include "reteco/biot_visco_solver.hpp"
#include <stdexcept>

namespace reteco {

BiotViscoStepSolver::BiotViscoStepSolver(
    const InvariantFEMMatrices2D& fem,
    const ParameterTheta& theta,
    const TimeGrid& grid)
    : fem_(fem),
      theta_(theta),
      grid_(grid),
      mem_beta_(theta.beta, grid, fem.num_disp_dofs()),
      is_uniform_grid_(grid.is_uniform()) {
    theta_.validate();
}

void BiotViscoStepSolver::reset(const Eigen::VectorXd& d0, const Eigen::VectorXd& p0) {
    if (static_cast<std::size_t>(d0.size()) != fem_.num_disp_dofs()) {
        throw std::invalid_argument("BiotViscoStepSolver::reset: d0 size mismatch.");
    }
    if (static_cast<std::size_t>(p0.size()) != fem_.num_press_dofs()) {
        throw std::invalid_argument("BiotViscoStepSolver::reset: p0 size mismatch.");
    }
    d_prev_ = d0;
    p_prev_ = p0;
    mem_beta_.reset(d0);
}

Eigen::SparseMatrix<double> BiotViscoStepSolver::build_saddle_matrix(
    double dt_n, double d_nn_beta) const {
    const auto& phys = fem_.physics();
    const std::size_t n_d = fem_.num_disp_dofs();
    const std::size_t n_p = fem_.num_press_dofs();
    const auto& is_dir    = fem_.is_dirichlet_disp();

    const double visco_Scale = phys.eta_v * std::pow(phys.tau_v, theta_.beta);
    const double eff_E       = theta_.E + visco_Scale * d_nn_beta;
    const double b           = phys.b;

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(
        static_cast<std::size_t>(fem_.K0_mech().nonZeros()) +
        2 * static_cast<std::size_t>(fem_.B_biot().nonZeros()) +
        static_cast<std::size_t>(fem_.M_p().nonZeros()) +
        static_cast<std::size_t>(fem_.K_p().nonZeros()));

    for (int k = 0; k < fem_.K0_mech().outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(fem_.K0_mech(), k); it; ++it) {
            const std::size_t r = static_cast<std::size_t>(it.row());
            const std::size_t c = static_cast<std::size_t>(it.col());
            if (is_dir[r] || is_dir[c]) {
                if (r == c) triplets.emplace_back(it.row(), it.col(), 1.0);
            } else {
                triplets.emplace_back(it.row(), it.col(), eff_E * it.value());
            }
        }
    }

    for (int k = 0; k < fem_.B_biot().outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(fem_.B_biot(), k); it; ++it) {
            const std::size_t r_d = static_cast<std::size_t>(it.row());
            const Eigen::Index c_p = static_cast<Eigen::Index>(n_d) + it.col();
            if (!is_dir[r_d]) {
                triplets.emplace_back(it.row(), c_p, -b * it.value());
                triplets.emplace_back(c_p, it.row(), +b * it.value());
            }
        }
    }

    const Eigen::SparseMatrix<double> A_pp =
        phys.S_p * fem_.M_p() + (dt_n * phys.K_h) * fem_.K_p();
    for (int k = 0; k < A_pp.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_pp, k); it; ++it) {
            triplets.emplace_back(
                static_cast<Eigen::Index>(n_d) + it.row(),
                static_cast<Eigen::Index>(n_d) + it.col(),
                it.value());
        }
    }

    Eigen::SparseMatrix<double> M_tot(
        static_cast<Eigen::Index>(n_d + n_p),
        static_cast<Eigen::Index>(n_d + n_p));
    M_tot.setFromTriplets(triplets.begin(), triplets.end());
    M_tot.makeCompressed();
    return M_tot;
}

void BiotViscoStepSolver::advance_step(
    std::size_t n,
    const Eigen::VectorXd& F_mech_n,
    const Eigen::VectorXd& S_fluid_n,
    Eigen::VectorXd& d_n_out,
    Eigen::VectorXd& p_n_out) {
    const auto& phys      = fem_.physics();
    const std::size_t n_d = fem_.num_disp_dofs();
    const std::size_t n_p = fem_.num_press_dofs();
    if (static_cast<std::size_t>(F_mech_n.size()) != n_d ||
        static_cast<std::size_t>(S_fluid_n.size()) != n_p) {
        throw std::invalid_argument("BiotViscoStepSolver::advance_step: forcing vector size mismatch.");
    }
    const auto& is_dir = fem_.is_dirichlet_disp();

    const double dt_n      = grid_.dt(n);
    const double d_nn_beta = mem_beta_.current_implicit_coeff(n);
    const Eigen::VectorXd H_beta = mem_beta_.compute_history_load(n);

    if (!is_uniform_grid_ || !factorized_) {
        const Eigen::SparseMatrix<double> A_saddle = build_saddle_matrix(dt_n, d_nn_beta);
        lu_solver_.compute(A_saddle);
        if (lu_solver_.info() != Eigen::Success) {
            throw std::runtime_error("BiotViscoStepSolver: SparseLU factorization failed.");
        }
        factorized_ = true;
    }

    const double visco_Scale = phys.eta_v * std::pow(phys.tau_v, theta_.beta);
    Eigen::VectorXd rhs_d = F_mech_n + visco_Scale * (fem_.K0_mech() * H_beta);
    for (std::size_t i = 0; i < n_d; ++i) {
        if (is_dir[i]) rhs_d(static_cast<Eigen::Index>(i)) = 0.0;
    }

    Eigen::VectorXd rhs_p =
        phys.S_p * (fem_.M_p() * p_prev_) +
        phys.b * (fem_.B_biot().transpose() * d_prev_) +
        dt_n * S_fluid_n;

    Eigen::VectorXd rhs_tot(static_cast<Eigen::Index>(n_d + n_p));
    rhs_tot.head(static_cast<Eigen::Index>(n_d)) = rhs_d;
    rhs_tot.tail(static_cast<Eigen::Index>(n_p)) = rhs_p;

    const Eigen::VectorXd sol = lu_solver_.solve(rhs_tot);
    if (lu_solver_.info() != Eigen::Success || !sol.allFinite()) {
        throw std::runtime_error("BiotViscoStepSolver: SparseLU solve failed.");
    }

    d_n_out = sol.head(static_cast<Eigen::Index>(n_d));
    p_n_out = sol.tail(static_cast<Eigen::Index>(n_p));

    mem_beta_.commit_step(n, d_n_out);
    d_prev_ = d_n_out;
    p_prev_ = p_n_out;
}

} // namespace reteco