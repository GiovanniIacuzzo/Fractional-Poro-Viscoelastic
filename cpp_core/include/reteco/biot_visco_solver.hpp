#pragma once
#include "reteco/invariant_matrices.hpp"
#include "reteco/caputo_l1_memory.hpp"
#include "reteco/parameters.hpp"
#include <Eigen/SparseLU>

namespace reteco {

/**
 * Risolve al passo temporale t_n il sottosistema accoppiato quasi-statico di Biot
 * con legge di Kelvin-Voigt frazionaria:
 *
 *   [ (E + eta_v*tau_v^beta*d_nn) K0_mech      -b * B_biot            ] [ d^n ]   [ rhs_d ]
 *   [  b * B_biot^T                        S_p * M_p + dt_n*K_h * K_p ] [ p^n ] = [ rhs_p ]
 */
class BiotViscoStepSolver {
public:
    BiotViscoStepSolver(
        const InvariantFEMMatrices2D& fem,
        const ParameterTheta& theta,
        const TimeGrid& grid);

    void reset(const Eigen::VectorXd& d0, const Eigen::VectorXd& p0);

    void advance_step(
        std::size_t n,
        const Eigen::VectorXd& F_mech_n,
        const Eigen::VectorXd& S_fluid_n,
        Eigen::VectorXd& d_n_out,
        Eigen::VectorXd& p_n_out);

private:
    const InvariantFEMMatrices2D& fem_;
    ParameterTheta theta_;
    TimeGrid grid_;
    CaputoL1Memory mem_beta_;

    Eigen::VectorXd d_prev_;
    Eigen::VectorXd p_prev_;

    bool is_uniform_grid_ = false;
    bool factorized_ = false;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu_solver_;

    Eigen::SparseMatrix<double> build_saddle_matrix(double dt_n, double d_nn_beta) const;
};

} // namespace reteco