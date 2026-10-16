#include "reteco/initial_compatibility.hpp"
#include <Eigen/SparseLU>
#include <stdexcept>
#include <cmath>

namespace reteco {

Eigen::VectorXd InitialCompatibility::compute_compatible_d0(
    const InvariantFEMMatrices2D& fem,
    double E_modulus,
    const Eigen::VectorXd& p0,
    const Eigen::VectorXd& F0) {
    const std::size_t n_d = fem.num_disp_dofs();
    const std::size_t n_p = fem.num_press_dofs();
    if (!std::isfinite(E_modulus) || E_modulus <= 0.0) {
        throw std::invalid_argument("InitialCompatibility: E_modulus must be finite and > 0.");
    }
    if (static_cast<std::size_t>(p0.size()) != n_p || static_cast<std::size_t>(F0.size()) != n_d) {
        throw std::invalid_argument("InitialCompatibility::compute_compatible_d0: input vector size mismatch.");
    }

    const auto& is_dir = fem.is_dirichlet_disp();
    const double b     = fem.physics().b;

    Eigen::VectorXd rhs = F0 + b * (fem.B_biot() * p0);
    Eigen::SparseMatrix<double> A = E_modulus * fem.K0_mech();

    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            const std::size_t r = static_cast<std::size_t>(it.row());
            const std::size_t c = static_cast<std::size_t>(it.col());
            if (is_dir[r] || is_dir[c]) {
                it.valueRef() = (r == c) ? 1.0 : 0.0;
            }
        }
    }
    for (std::size_t i = 0; i < n_d; ++i) {
        if (is_dir[i]) {
            rhs(static_cast<Eigen::Index>(i)) = 0.0;
        }
    }

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("InitialCompatibility: LU factorization of elastic matrix failed.");
    }
    Eigen::VectorXd d0 = solver.solve(rhs);
    if (solver.info() != Eigen::Success || !d0.allFinite()) {
        throw std::runtime_error("InitialCompatibility: LU solve of initial displacement failed.");
    }
    return d0;
}

double InitialCompatibility::compatibility_residual_norm(
    const InvariantFEMMatrices2D& fem,
    double E_modulus,
    const Eigen::VectorXd& d0,
    const Eigen::VectorXd& p0,
    const Eigen::VectorXd& F0) {
    if (static_cast<std::size_t>(d0.size()) != fem.num_disp_dofs() ||
        static_cast<std::size_t>(p0.size()) != fem.num_press_dofs() ||
        static_cast<std::size_t>(F0.size()) != fem.num_disp_dofs()) {
        throw std::invalid_argument("InitialCompatibility::compatibility_residual_norm: vector size mismatch.");
    }
    const auto& is_dir = fem.is_dirichlet_disp();
    const double b     = fem.physics().b;

    Eigen::VectorXd res = (E_modulus * (fem.K0_mech() * d0)) - b * (fem.B_biot() * p0) - F0;
    for (std::size_t i = 0; i < fem.num_disp_dofs(); ++i) {
        if (is_dir[i]) {
            res(static_cast<Eigen::Index>(i)) = 0.0;
        }
    }
    return res.norm();
}

} // namespace reteco