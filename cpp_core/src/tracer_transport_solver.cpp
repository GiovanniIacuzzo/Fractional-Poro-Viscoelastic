#include "reteco/tracer_transport_solver.hpp"
#include <Eigen/SparseLU>
#include <stdexcept>

namespace reteco {

TracerTransportStepSolver::TracerTransportStepSolver(
    const InvariantFEMMatrices2D& fem,
    const ParameterTheta& theta,
    const TimeGrid& grid,
    bool use_supg)
    : fem_(fem),
      theta_(theta),
      grid_(grid),
      mem_alpha_(theta.alpha, grid, fem.num_tracer_dofs()),
      use_supg_(use_supg) {
    theta_.validate();
}

void TracerTransportStepSolver::reset(const Eigen::VectorXd& c0) {
    mem_alpha_.reset(c0);
}

void TracerTransportStepSolver::advance_step(
    std::size_t n,
    const Eigen::VectorXd& p_n,
    const Eigen::VectorXd& S_tracer_n,
    Eigen::VectorXd& c_n_out) {
    const auto& phys = fem_.physics();

    const double d_nn_alpha       = mem_alpha_.current_implicit_coeff(n);
    const Eigen::VectorXd H_alpha = mem_alpha_.compute_history_load(n);

    // Con T_0 == tau_c, l'equazione adimensionale del tracciante è (Sezione 2 Contratto Matematico):
    // phi_0 * (CD_t^alpha c^n, z) - (c^n * q^n, grad(z)) + phi_0 * D * (grad(c^n), grad(z)) = (s_c^n, z)
    const Eigen::SparseMatrix<double> A_adv =
        fem_.assemble_tracer_advection(p_n, use_supg_, theta_.D);

    Eigen::SparseMatrix<double> A_c =
        (phys.phi_0 * d_nn_alpha) * fem_.M_c() +
        (phys.phi_0 * theta_.D) * fem_.K_c() +
        A_adv;
    A_c.makeCompressed();

    const Eigen::VectorXd rhs_c = phys.phi_0 * (fem_.M_c() * H_alpha) + S_tracer_n;

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_c);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("TracerTransportStepSolver: SparseLU factorization failed.");
    }
    c_n_out = solver.solve(rhs_c);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("TracerTransportStepSolver: SparseLU solve failed.");
    }

    mem_alpha_.commit_step(n, c_n_out);
}

} // namespace reteco