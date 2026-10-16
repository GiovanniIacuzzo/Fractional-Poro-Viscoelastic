#include "reteco/invariant_matrices.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <map>

namespace reteco {

InvariantFEMMatrices2D::InvariantFEMMatrices2D(
    unsigned int subdivisions_N,
    const FixedPhysicsConfig& physics,
    unsigned int fe_deg_disp,
    unsigned int fe_deg_press,
    unsigned int fe_deg_tracer,
    bool dirichlet_all_displacement_boundary)
    : physics_(physics),
      dirichlet_all_disp_boundary_(dirichlet_all_displacement_boundary),
      fe_disp_(dealii::FE_Q<2>(fe_deg_disp), 2),
      fe_press_(fe_deg_press),
      fe_tracer_(fe_deg_tracer),
      dh_disp_(tria_),
      dh_press_(tria_),
      dh_tracer_(tria_) {
    physics_.validate();
    setup_grid_and_dofs(subdivisions_N);
    assemble_all_invariant();
}

void InvariantFEMMatrices2D::setup_grid_and_dofs(unsigned int subdivisions_N) {
    // colorize = true assegna boundary_id: 0=left(x=0), 1=right(x=1), 2=bottom(y=0), 3=top(y=1)
    dealii::GridGenerator::subdivided_hyper_cube(tria_, subdivisions_N, 0.0, 1.0, true);

    dh_disp_.distribute_dofs(fe_disp_);
    dh_press_.distribute_dofs(fe_press_);
    dh_tracer_.distribute_dofs(fe_tracer_);

    n_d_ = dh_disp_.n_dofs();
    n_p_ = dh_press_.n_dofs();
    n_c_ = dh_tracer_.n_dofs();

    is_dirichlet_disp_.assign(n_d_, false);
    std::map<dealii::types::global_dof_index, double> bvals;

    if (dirichlet_all_disp_boundary_) {
        for (dealii::types::boundary_id bid = 0; bid < 4; ++bid) {
            dealii::VectorTools::interpolate_boundary_values(
                dh_disp_, bid, dealii::Functions::ZeroFunction<2>(2), bvals);
        }
    } else {
        // Solo Gamma_D = {x = 0} (boundary_id 0) vincolato, resto Gamma_N libero per trazioni
        dealii::VectorTools::interpolate_boundary_values(
            dh_disp_, 0, dealii::Functions::ZeroFunction<2>(2), bvals);
    }

    for (const auto& kv : bvals) {
        is_dirichlet_disp_[kv.first] = true;
    }
}

void InvariantFEMMatrices2D::assemble_all_invariant() {
    const unsigned int quad_deg =
        std::max({fe_disp_.degree, fe_press_.degree, fe_tracer_.degree}) + 2;
    dealii::QGauss<2> quadrature(quad_deg);

    dealii::FEValues<2> fe_val_d(
        fe_disp_, quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    dealii::FEValues<2> fe_val_p(
        fe_press_, quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    dealii::FEValues<2> fe_val_c(
        fe_tracer_, quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int dofs_per_cell_d = fe_disp_.n_dofs_per_cell();
    const unsigned int dofs_per_cell_p = fe_press_.n_dofs_per_cell();
    const unsigned int dofs_per_cell_c = fe_tracer_.n_dofs_per_cell();
    const unsigned int n_q_points      = quadrature.size();

    const double two_mu0 = 2.0 * physics_.mu_0();
    const double lam0    = physics_.lambda_0();

    std::vector<Eigen::Triplet<double>> trip_K0;
    std::vector<Eigen::Triplet<double>> trip_B;
    std::vector<Eigen::Triplet<double>> trip_Mp;
    std::vector<Eigen::Triplet<double>> trip_Kp;
    std::vector<Eigen::Triplet<double>> trip_Mc;
    std::vector<Eigen::Triplet<double>> trip_Kc;

    const dealii::FEValuesExtractors::Vector disp_ext(0);

    auto cell_d = dh_disp_.begin_active();
    auto cell_p = dh_press_.begin_active();
    auto cell_c = dh_tracer_.begin_active();

    std::vector<dealii::types::global_dof_index> loc_d(dofs_per_cell_d);
    std::vector<dealii::types::global_dof_index> loc_p(dofs_per_cell_p);
    std::vector<dealii::types::global_dof_index> loc_c(dofs_per_cell_c);

    for (; cell_d != dh_disp_.end(); ++cell_d, ++cell_p, ++cell_c) {
        fe_val_d.reinit(cell_d);
        fe_val_p.reinit(cell_p);
        fe_val_c.reinit(cell_c);

        cell_d->get_dof_indices(loc_d);
        cell_p->get_dof_indices(loc_p);
        cell_c->get_dof_indices(loc_c);

        Eigen::MatrixXd loc_K0 = Eigen::MatrixXd::Zero(dofs_per_cell_d, dofs_per_cell_d);
        Eigen::MatrixXd loc_B  = Eigen::MatrixXd::Zero(dofs_per_cell_d, dofs_per_cell_p);
        Eigen::MatrixXd loc_Mp = Eigen::MatrixXd::Zero(dofs_per_cell_p, dofs_per_cell_p);
        Eigen::MatrixXd loc_Kp = Eigen::MatrixXd::Zero(dofs_per_cell_p, dofs_per_cell_p);
        Eigen::MatrixXd loc_Mc = Eigen::MatrixXd::Zero(dofs_per_cell_c, dofs_per_cell_c);
        Eigen::MatrixXd loc_Kc = Eigen::MatrixXd::Zero(dofs_per_cell_c, dofs_per_cell_c);

        for (unsigned int q = 0; q < n_q_points; ++q) {
            const double JxW = fe_val_d.JxW(q);

            // 1. Meccanica K0_mech e accoppiamento B_biot
            for (unsigned int i = 0; i < dofs_per_cell_d; ++i) {
                const auto eps_i = fe_val_d[disp_ext].symmetric_gradient(i, q);
                const double div_i = dealii::trace(eps_i);

                for (unsigned int j = 0; j < dofs_per_cell_d; ++j) {
                    const auto eps_j = fe_val_d[disp_ext].symmetric_gradient(j, q);
                    const double div_j = dealii::trace(eps_j);
                    const double c0_colon = two_mu0 * (eps_i * eps_j) + lam0 * div_i * div_j;
                    loc_K0(i, j) += c0_colon * JxW;
                }

                for (unsigned int j = 0; j < dofs_per_cell_p; ++j) {
                    const double phi_p_j = fe_val_p.shape_value(j, q);
                    loc_B(i, j) += div_i * phi_p_j * JxW;
                }
            }

            // 2. Pressione M_p e K_p
            for (unsigned int i = 0; i < dofs_per_cell_p; ++i) {
                const double w_i    = fe_val_p.shape_value(i, q);
                const auto grad_w_i = fe_val_p.shape_grad(i, q);
                for (unsigned int j = 0; j < dofs_per_cell_p; ++j) {
                    const double p_j    = fe_val_p.shape_value(j, q);
                    const auto grad_p_j = fe_val_p.shape_grad(j, q);
                    loc_Mp(i, j) += w_i * p_j * JxW;
                    loc_Kp(i, j) += (grad_w_i * grad_p_j) * JxW;
                }
            }

            // 3. Tracciante M_c e K_c
            for (unsigned int i = 0; i < dofs_per_cell_c; ++i) {
                const double z_i    = fe_val_c.shape_value(i, q);
                const auto grad_z_i = fe_val_c.shape_grad(i, q);
                for (unsigned int j = 0; j < dofs_per_cell_c; ++j) {
                    const double c_j    = fe_val_c.shape_value(j, q);
                    const auto grad_c_j = fe_val_c.shape_grad(j, q);
                    loc_Mc(i, j) += z_i * c_j * JxW;
                    loc_Kc(i, j) += (grad_z_i * grad_c_j) * JxW;
                }
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell_d; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell_d; ++j) {
                if (std::abs(loc_K0(i, j)) > 1.0e-16)
                    trip_K0.emplace_back(loc_d[i], loc_d[j], loc_K0(i, j));
            }
            for (unsigned int j = 0; j < dofs_per_cell_p; ++j) {
                if (std::abs(loc_B(i, j)) > 1.0e-16)
                    trip_B.emplace_back(loc_d[i], loc_p[j], loc_B(i, j));
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell_p; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell_p; ++j) {
                if (std::abs(loc_Mp(i, j)) > 1.0e-16)
                    trip_Mp.emplace_back(loc_p[i], loc_p[j], loc_Mp(i, j));
                if (std::abs(loc_Kp(i, j)) > 1.0e-16)
                    trip_Kp.emplace_back(loc_p[i], loc_p[j], loc_Kp(i, j));
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell_c; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell_c; ++j) {
                if (std::abs(loc_Mc(i, j)) > 1.0e-16)
                    trip_Mc.emplace_back(loc_c[i], loc_c[j], loc_Mc(i, j));
                if (std::abs(loc_Kc(i, j)) > 1.0e-16)
                    trip_Kc.emplace_back(loc_c[i], loc_c[j], loc_Kc(i, j));
            }
        }
    }

    K0_mech_.resize(static_cast<Eigen::Index>(n_d_), static_cast<Eigen::Index>(n_d_));
    B_biot_.resize(static_cast<Eigen::Index>(n_d_), static_cast<Eigen::Index>(n_p_));
    M_p_.resize(static_cast<Eigen::Index>(n_p_), static_cast<Eigen::Index>(n_p_));
    K_p_.resize(static_cast<Eigen::Index>(n_p_), static_cast<Eigen::Index>(n_p_));
    M_c_.resize(static_cast<Eigen::Index>(n_c_), static_cast<Eigen::Index>(n_c_));
    K_c_.resize(static_cast<Eigen::Index>(n_c_), static_cast<Eigen::Index>(n_c_));

    K0_mech_.setFromTriplets(trip_K0.begin(), trip_K0.end());
    B_biot_.setFromTriplets(trip_B.begin(), trip_B.end());
    M_p_.setFromTriplets(trip_Mp.begin(), trip_Mp.end());
    K_p_.setFromTriplets(trip_Kp.begin(), trip_Kp.end());
    M_c_.setFromTriplets(trip_Mc.begin(), trip_Mc.end());
    K_c_.setFromTriplets(trip_Kc.begin(), trip_Kc.end());

    K0_mech_.makeCompressed();
    B_biot_.makeCompressed();
    M_p_.makeCompressed();
    K_p_.makeCompressed();
    M_c_.makeCompressed();
    K_c_.makeCompressed();
}

Eigen::SparseMatrix<double> InvariantFEMMatrices2D::assemble_tracer_advection(
    const Eigen::VectorXd& p_dofs,
    bool supg_stabilization,
    double D_tracer) const {
    const unsigned int quad_deg = std::max(fe_press_.degree, fe_tracer_.degree) + 2;
    dealii::QGauss<2> quadrature(quad_deg);

    dealii::FEValues<2> fe_val_p(fe_press_, quadrature, dealii::update_gradients);
    dealii::FEValues<2> fe_val_c(
        fe_tracer_, quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int dofs_p = fe_press_.n_dofs_per_cell();
    const unsigned int dofs_c = fe_tracer_.n_dofs_per_cell();
    const unsigned int n_q    = quadrature.size();

    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<dealii::types::global_dof_index> loc_p(dofs_p);
    std::vector<dealii::types::global_dof_index> loc_c(dofs_c);

    auto cell_p = dh_press_.begin_active();
    auto cell_c = dh_tracer_.begin_active();

    for (; cell_c != dh_tracer_.end(); ++cell_p, ++cell_c) {
        fe_val_p.reinit(cell_p);
        fe_val_c.reinit(cell_c);
        cell_p->get_dof_indices(loc_p);
        cell_c->get_dof_indices(loc_c);

        const double h_K = cell_c->diameter();
        Eigen::MatrixXd loc_adv = Eigen::MatrixXd::Zero(dofs_c, dofs_c);

        for (unsigned int q = 0; q < n_q; ++q) {
            const double JxW = fe_val_c.JxW(q);

            dealii::Tensor<1, 2> grad_p_q;
            for (unsigned int k = 0; k < dofs_p; ++k) {
                grad_p_q += p_dofs(static_cast<Eigen::Index>(loc_p[k])) * fe_val_p.shape_grad(k, q);
            }
            const dealii::Tensor<1, 2> q_darcy = -physics_.K_h * grad_p_q;
            const double q_norm = q_darcy.norm();

            double tau_supg = 0.0;
            if (supg_stabilization && q_norm > 1.0e-12 && D_tracer > 0.0) {
                const double diff_eff = physics_.phi_0 * D_tracer;
                const double Pe_K = (q_norm * h_K) / (2.0 * diff_eff);
                if (Pe_K > 1.0) {
                    tau_supg = (h_K / (2.0 * q_norm)) * (1.0 - 1.0 / Pe_K);
                }
            }

            for (unsigned int i = 0; i < dofs_c; ++i) {
                const auto grad_z_i = fe_val_c.shape_grad(i, q);
                const double q_dot_grad_z = q_darcy * grad_z_i;

                for (unsigned int j = 0; j < dofs_c; ++j) {
                    const double c_j = fe_val_c.shape_value(j, q);
                    double val = -(c_j * q_dot_grad_z) * JxW;

                    if (tau_supg > 0.0) {
                        const auto grad_c_j = fe_val_c.shape_grad(j, q);
                        val += tau_supg * (q_darcy * grad_c_j) * q_dot_grad_z * JxW;
                    }
                    loc_adv(i, j) += val;
                }
            }
        }

        for (unsigned int i = 0; i < dofs_c; ++i) {
            for (unsigned int j = 0; j < dofs_c; ++j) {
                if (std::abs(loc_adv(i, j)) > 1.0e-16) {
                    triplets.emplace_back(loc_c[i], loc_c[j], loc_adv(i, j));
                }
            }
        }
    }

    Eigen::SparseMatrix<double> A_adv(
        static_cast<Eigen::Index>(n_c_), static_cast<Eigen::Index>(n_c_));
    A_adv.setFromTriplets(triplets.begin(), triplets.end());
    A_adv.makeCompressed();
    return A_adv;
}

double InvariantFEMMatrices2D::compute_max_darcy_velocity(const Eigen::VectorXd& p_dofs) const {
    dealii::QGauss<2> quadrature(fe_press_.degree + 2);
    dealii::FEValues<2> fe_val_p(fe_press_, quadrature, dealii::update_gradients);
    const unsigned int dofs_p = fe_press_.n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> loc_p(dofs_p);

    double max_q = 0.0;
    for (const auto& cell_p : dh_press_.active_cell_iterators()) {
        fe_val_p.reinit(cell_p);
        cell_p->get_dof_indices(loc_p);
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            dealii::Tensor<1, 2> grad_p_q;
            for (unsigned int k = 0; k < dofs_p; ++k) {
                grad_p_q += p_dofs(static_cast<Eigen::Index>(loc_p[k])) * fe_val_p.shape_grad(k, q);
            }
            const double q_norm = (physics_.K_h * grad_p_q).norm();
            if (q_norm > max_q) max_q = q_norm;
        }
    }
    return max_q;
}

static Eigen::MatrixXd extract_support_points_helper(const dealii::DoFHandler<2>& dh) {
    const std::map<dealii::types::global_dof_index, dealii::Point<2>> support_points =
        dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<2>(), dh);
    Eigen::MatrixXd pts(static_cast<Eigen::Index>(dh.n_dofs()), 2);
    for (const auto& [dof_idx, pt] : support_points) {
        pts(static_cast<Eigen::Index>(dof_idx), 0) = pt[0];
        pts(static_cast<Eigen::Index>(dof_idx), 1) = pt[1];
    }
    return pts;
}

Eigen::MatrixXd InvariantFEMMatrices2D::get_disp_support_points() const {
    return extract_support_points_helper(dh_disp_);
}

Eigen::MatrixXd InvariantFEMMatrices2D::get_press_support_points() const {
    return extract_support_points_helper(dh_press_);
}

Eigen::MatrixXd InvariantFEMMatrices2D::get_tracer_support_points() const {
    return extract_support_points_helper(dh_tracer_);
}

} // namespace reteco