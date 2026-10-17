#pragma once
#include "reteco/parameters.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>

#include <Eigen/Sparse>
#include <vector>

namespace reteco {

/**
 * Gestisce gli spazi agli elementi finiti stabili in deal.II su dominio 2D:
 * - Spostamento d_h in [Q_2]^2 (Taylor-Hood)
 * - Pressione p_h in Q_1       (Taylor-Hood)
 * - Concentrazione c_h in Q_1 (o Q_2)
 *
 * Pre-assembla una sola volta tutte le matrici invarianti rispetto a theta = (alpha, beta, D, E):
 * - K0_mech : int_Omega (C_0(nu):eps(u)) : eps(v) dx
 * - B_biot  : int_Omega p * div(v) dx
 * - M_p     : int_Omega p * w dx
 * - K_p     : int_Omega grad(p) . grad(w) dx
 * - M_c     : int_Omega c * z dx
 * - K_c     : int_Omega grad(c) . grad(z) dx
 */
class InvariantFEMMatrices2D {
public:
    InvariantFEMMatrices2D(
        unsigned int subdivisions_N,
        const FixedPhysicsConfig& physics,
        unsigned int fe_deg_disp = 2,
        unsigned int fe_deg_press = 1,
        unsigned int fe_deg_tracer = 1,
        bool dirichlet_all_displacement_boundary = true);

    ~InvariantFEMMatrices2D() {
        dh_disp_.clear();
        dh_press_.clear();
        dh_tracer_.clear();
    }

    std::size_t num_disp_dofs() const { return n_d_; }
    std::size_t num_press_dofs() const { return n_p_; }
    std::size_t num_tracer_dofs() const { return n_c_; }

    const Eigen::SparseMatrix<double>& K0_mech() const { return K0_mech_; }
    const Eigen::SparseMatrix<double>& B_biot() const { return B_biot_; }
    const Eigen::SparseMatrix<double>& M_p() const { return M_p_; }
    const Eigen::SparseMatrix<double>& K_p() const { return K_p_; }
    const Eigen::SparseMatrix<double>& M_c() const { return M_c_; }
    const Eigen::SparseMatrix<double>& K_c() const { return K_c_; }

    const std::vector<bool>& is_dirichlet_disp() const { return is_dirichlet_disp_; }
    const FixedPhysicsConfig& physics() const { return physics_; }

    const dealii::Triangulation<2>& triangulation() const { return tria_; }
    const dealii::DoFHandler<2>& dof_handler_disp() const { return dh_disp_; }
    const dealii::DoFHandler<2>& dof_handler_press() const { return dh_press_; }
    const dealii::DoFHandler<2>& dof_handler_tracer() const { return dh_tracer_; }

    // Assembla la matrice advettiva dipendente dal campo di pressione p_n:
    // A_adv(i, j) = - int_Omega (phi_j * q_h) . grad(phi_i) dx, con q_h = -K_h * grad(p_h)
    Eigen::SparseMatrix<double> assemble_tracer_advection(
        const Eigen::VectorXd& p_dofs,
        bool supg_stabilization = false,
        double D_tracer = 0.0) const;

    // Calcola la norma L^\infty(Omega) di div(q_h) o |q_h| per diagnostica
    double compute_max_darcy_velocity(const Eigen::VectorXd& p_dofs) const;

    // Coordinate dei punti di supporto dei gradi di liberta'
    Eigen::MatrixXd get_disp_support_points() const;
    Eigen::MatrixXd get_press_support_points() const;
    Eigen::MatrixXd get_tracer_support_points() const;

private:
    FixedPhysicsConfig physics_;
    bool dirichlet_all_disp_boundary_;

    dealii::Triangulation<2> tria_;
    dealii::FESystem<2> fe_disp_;
    dealii::FE_Q<2> fe_press_;
    dealii::FE_Q<2> fe_tracer_;

    dealii::DoFHandler<2> dh_disp_;
    dealii::DoFHandler<2> dh_press_;
    dealii::DoFHandler<2> dh_tracer_;

    std::size_t n_d_ = 0;
    std::size_t n_p_ = 0;
    std::size_t n_c_ = 0;

    std::vector<bool> is_dirichlet_disp_;

    Eigen::SparseMatrix<double> K0_mech_;
    Eigen::SparseMatrix<double> B_biot_;
    Eigen::SparseMatrix<double> M_p_;
    Eigen::SparseMatrix<double> K_p_;
    Eigen::SparseMatrix<double> M_c_;
    Eigen::SparseMatrix<double> K_c_;

    void setup_grid_and_dofs(unsigned int subdivisions_N);
    void assemble_all_invariant();
};

} // namespace reteco