#include "reteco/mms_manufactured.hpp"
#include "reteco/biot_visco_solver.hpp"
#include "reteco/tracer_transport_solver.hpp"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <cmath>
#include <numbers>

namespace reteco {

namespace {
constexpr double PI = std::numbers::pi;
}

MMSManufacturedSystem::MMSManufacturedSystem(
    const FixedPhysicsConfig& physics,
    const ParameterTheta& theta,
    double delta_tracer)
    : physics_(physics),
      theta_(theta),
      delta_(delta_tracer),
      caputo_coeff_beta_(2.0 / std::tgamma(3.0 - theta.beta)),
      caputo_coeff_alpha_(2.0 / std::tgamma(3.0 - theta.alpha)) {
    physics_.validate();
    theta_.validate();
    if (!(delta_ > 0.0 && delta_ < 1.0)) {
        throw std::invalid_argument("MMSManufacturedSystem: delta_tracer must be in (0, 1).");
    }
}

Eigen::Vector2d MMSManufacturedSystem::exact_displacement(double x, double y, double t) const {
    const double t2 = t * t;
    return Eigen::Vector2d(
        t2 * std::sin(PI * x) * std::sin(PI * y),
        t2 * std::sin(2.0 * PI * x) * std::sin(PI * y));
}

Eigen::Matrix2d MMSManufacturedSystem::exact_displacement_grad(double x, double y, double t) const {
    const double t2 = t * t;
    Eigen::Matrix2d G;
    G(0, 0) = t2 * PI * std::cos(PI * x) * std::sin(PI * y);
    G(0, 1) = t2 * PI * std::sin(PI * x) * std::cos(PI * y);
    G(1, 0) = t2 * 2.0 * PI * std::cos(2.0 * PI * x) * std::sin(PI * y);
    G(1, 1) = t2 * PI * std::sin(2.0 * PI * x) * std::cos(PI * y);
    return G;
}

double MMSManufacturedSystem::exact_pressure(double x, double y, double t) const {
    return (t * t) * std::cos(PI * x) * std::cos(PI * y);
}

Eigen::Vector2d MMSManufacturedSystem::exact_pressure_grad(double x, double y, double t) const {
    const double t2 = t * t;
    return Eigen::Vector2d(
        -PI * t2 * std::sin(PI * x) * std::cos(PI * y),
        -PI * t2 * std::cos(PI * x) * std::sin(PI * y));
}

double MMSManufacturedSystem::exact_tracer(double x, double y, double t) const {
    return 1.0 + delta_ * (t * t) * std::cos(PI * x) * std::cos(PI * y);
}

Eigen::Vector2d MMSManufacturedSystem::exact_tracer_grad(double x, double y, double t) const {
    return delta_ * exact_pressure_grad(x, y, t);
}

Eigen::Vector2d MMSManufacturedSystem::exact_f_s(double x, double y, double t) const {
    const double t2      = t * t;
    const double dt_beta = caputo_coeff_beta_ * std::pow(t, 2.0 - theta_.beta);
    const double A_mod   = theta_.E * t2 + physics_.eta_v * std::pow(physics_.tau_v, theta_.beta) * dt_beta;

    const double mu0  = physics_.mu_0();
    const double lam0 = physics_.lambda_0();

    // u1 = sin(pi x) sin(pi y), u2 = sin(2 pi x) sin(pi y)
    const double u1_xx = -(PI * PI) * std::sin(PI * x) * std::sin(PI * y);
    const double u1_yy = -(PI * PI) * std::sin(PI * x) * std::sin(PI * y);
    const double u1_xy = +(PI * PI) * std::cos(PI * x) * std::cos(PI * y);

    const double u2_xx = -4.0 * (PI * PI) * std::sin(2.0 * PI * x) * std::sin(PI * y);
    const double u2_yy = -(PI * PI) * std::sin(2.0 * PI * x) * std::sin(PI * y);
    const double u2_xy = +2.0 * (PI * PI) * std::cos(2.0 * PI * x) * std::cos(PI * y);

    // div(C0 : eps(u)) = mu0 * Lap(u) + (mu0 + lam0) * grad(div(u))
    const double div_sig_s_x = A_mod * (
        (2.0 * mu0 + lam0) * u1_xx + mu0 * u1_yy + (mu0 + lam0) * u2_xy);
    const double div_sig_s_y = A_mod * (
        mu0 * u2_xx + (2.0 * mu0 + lam0) * u2_yy + (mu0 + lam0) * u1_xy);

    const Eigen::Vector2d grad_p = exact_pressure_grad(x, y, t);
    // f_s* = -div(sigma*) = -div(sigma_s*) + b * grad(p*)
    return Eigen::Vector2d(
        -div_sig_s_x + physics_.b * grad_p.x(),
        -div_sig_s_y + physics_.b * grad_p.y());
}

double MMSManufacturedSystem::exact_s_f(double x, double y, double t) const {
    const double t2  = t * t;
    const double dt2 = 2.0 * t;

    const double P = std::cos(PI * x) * std::cos(PI * y);
    const double div_u_spatial =
        PI * std::cos(PI * x) * std::sin(PI * y) +
        PI * std::sin(2.0 * PI * x) * std::cos(PI * y);
    const double lap_P = -2.0 * (PI * PI) * P;

    // s_f* = S_p * dp*/dt + b * d(div d*)/dt - K_h * Lap(p*)
    return physics_.S_p * dt2 * P +
           physics_.b * dt2 * div_u_spatial -
           physics_.K_h * t2 * lap_P;
}

double MMSManufacturedSystem::exact_s_c(double x, double y, double t) const {
    const double t2       = t * t;
    const double dt_alpha = caputo_coeff_alpha_ * std::pow(t, 2.0 - theta_.alpha);

    const double P     = std::cos(PI * x) * std::cos(PI * y);
    const double lap_P = -2.0 * (PI * PI) * P;
    const double grad_P_sq =
        (PI * PI) * (std::pow(std::sin(PI * x) * std::cos(PI * y), 2) +
                     std::pow(std::cos(PI * x) * std::sin(PI * y), 2));

    const double c_val = 1.0 + delta_ * t2 * P;
    // div(c* q*) = grad(c*).q* + c* div(q*)
    // con q* = -K_h * t^2 * grad(P) e grad(c*) = delta * t^2 * grad(P):
    const double grad_c_dot_q = -physics_.K_h * delta_ * (t2 * t2) * grad_P_sq;
    const double c_div_q      = -physics_.K_h * t2 * c_val * lap_P;
    const double div_cq       = grad_c_dot_q + c_div_q;

    const double diff_term = -physics_.phi_0 * theta_.D * delta_ * t2 * lap_P;
    const double caputo_c  = physics_.phi_0 * delta_ * dt_alpha * P;

    return caputo_c + div_cq + diff_term;
}

Eigen::VectorXd MMSManufacturedSystem::assemble_F_mech_rhs(
    const InvariantFEMMatrices2D& fem, double t) const {
    dealii::QGauss<2> quadrature(fem.dof_handler_disp().get_fe().degree + 3);
    dealii::FEValues<2> fe_val(
        fem.dof_handler_disp().get_fe(), quadrature,
        dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fem.dof_handler_disp().get_fe().n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> loc_dofs(dofs_per_cell);
    const dealii::FEValuesExtractors::Vector disp_ext(0);

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(fem.num_disp_dofs()));

    for (const auto& cell : fem.dof_handler_disp().active_cell_iterators()) {
        fe_val.reinit(cell);
        cell->get_dof_indices(loc_dofs);
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            const auto pt = fe_val.quadrature_point(q);
            const Eigen::Vector2d fs = exact_f_s(pt[0], pt[1], t);
            dealii::Tensor<1, 2> fs_tensor;
            fs_tensor[0] = fs.x();
            fs_tensor[1] = fs.y();
            const double JxW = fe_val.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const auto v_i = fe_val[disp_ext].value(i, q);
                rhs(static_cast<Eigen::Index>(loc_dofs[i])) += (fs_tensor * v_i) * JxW;
            }
        }
    }
    return rhs;
}

Eigen::VectorXd MMSManufacturedSystem::assemble_S_fluid_rhs(
    const InvariantFEMMatrices2D& fem, double t) const {
    dealii::QGauss<2> quadrature(fem.dof_handler_press().get_fe().degree + 3);
    dealii::FEValues<2> fe_val(
        fem.dof_handler_press().get_fe(), quadrature,
        dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fem.dof_handler_press().get_fe().n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> loc_dofs(dofs_per_cell);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(fem.num_press_dofs()));

    for (const auto& cell : fem.dof_handler_press().active_cell_iterators()) {
        fe_val.reinit(cell);
        cell->get_dof_indices(loc_dofs);
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            const auto pt   = fe_val.quadrature_point(q);
            const double sf = exact_s_f(pt[0], pt[1], t);
            const double JxW = fe_val.JxW(q);
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                rhs(static_cast<Eigen::Index>(loc_dofs[i])) += sf * fe_val.shape_value(i, q) * JxW;
            }
        }
    }
    return rhs;
}

Eigen::VectorXd MMSManufacturedSystem::assemble_S_tracer_rhs(
    const InvariantFEMMatrices2D& fem, double t) const {
    dealii::QGauss<2> quadrature(fem.dof_handler_tracer().get_fe().degree + 3);
    dealii::FEValues<2> fe_val(
        fem.dof_handler_tracer().get_fe(), quadrature,
        dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fem.dof_handler_tracer().get_fe().n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> loc_dofs(dofs_per_cell);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(fem.num_tracer_dofs()));

    for (const auto& cell : fem.dof_handler_tracer().active_cell_iterators()) {
        fe_val.reinit(cell);
        cell->get_dof_indices(loc_dofs);
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            const auto pt   = fe_val.quadrature_point(q);
            const double sc = exact_s_c(pt[0], pt[1], t);
            const double JxW = fe_val.JxW(q);
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                rhs(static_cast<Eigen::Index>(loc_dofs[i])) += sc * fe_val.shape_value(i, q) * JxW;
            }
        }
    }
    return rhs;
}

Eigen::VectorXd MMSManufacturedSystem::project_initial_tracer(
    const InvariantFEMMatrices2D& fem) const {
    // Poiche' c*(x,y,0) = 1.0 e le funzioni di forma di Lagrange sommano a 1:
    return Eigen::VectorXd::Ones(static_cast<Eigen::Index>(fem.num_tracer_dofs()));
}

MMSErrorReport MMSManufacturedSystem::compute_fem_errors(
    const InvariantFEMMatrices2D& fem,
    double t,
    const Eigen::VectorXd& d_h,
    const Eigen::VectorXd& p_h,
    const Eigen::VectorXd& c_h,
    double dt_max) const {
    dealii::QGauss<2> quadrature(5);

    dealii::FEValues<2> fe_val_d(
        fem.dof_handler_disp().get_fe(), quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<2> fe_val_p(
        fem.dof_handler_press().get_fe(), quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    dealii::FEValues<2> fe_val_c(
        fem.dof_handler_tracer().get_fe(), quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int dofs_d = fem.dof_handler_disp().get_fe().n_dofs_per_cell();
    const unsigned int dofs_p = fem.dof_handler_press().get_fe().n_dofs_per_cell();
    const unsigned int dofs_c = fem.dof_handler_tracer().get_fe().n_dofs_per_cell();

    std::vector<dealii::types::global_dof_index> loc_d(dofs_d);
    std::vector<dealii::types::global_dof_index> loc_p(dofs_p);
    std::vector<dealii::types::global_dof_index> loc_c(dofs_c);

    const dealii::FEValuesExtractors::Vector disp_ext(0);

    double d_l2_sq = 0.0, d_h1_sq = 0.0;
    double p_l2_sq = 0.0, p_h1_sq = 0.0;
    double c_l2_sq = 0.0, c_h1_sq = 0.0;
    double h_max   = 0.0;

    auto cell_d = fem.dof_handler_disp().begin_active();
    auto cell_p = fem.dof_handler_press().begin_active();
    auto cell_c = fem.dof_handler_tracer().begin_active();

    for (; cell_d != fem.dof_handler_disp().end(); ++cell_d, ++cell_p, ++cell_c) {
        fe_val_d.reinit(cell_d);
        fe_val_p.reinit(cell_p);
        fe_val_c.reinit(cell_c);

        cell_d->get_dof_indices(loc_d);
        cell_p->get_dof_indices(loc_p);
        cell_c->get_dof_indices(loc_c);

        h_max = std::max(h_max, cell_d->diameter());

        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            const auto pt    = fe_val_d.quadrature_point(q);
            const double x   = pt[0];
            const double y   = pt[1];
            const double JxW = fe_val_d.JxW(q);

            // Spostamento
            dealii::Tensor<1, 2> d_num;
            dealii::Tensor<2, 2> grad_d_num;
            for (unsigned int k = 0; k < dofs_d; ++k) {
                const double val_k = d_h(static_cast<Eigen::Index>(loc_d[k]));
                d_num      += val_k * fe_val_d[disp_ext].value(k, q);
                grad_d_num += val_k * fe_val_d[disp_ext].gradient(k, q);
            }
            const Eigen::Vector2d d_ex      = exact_displacement(x, y, t);
            const Eigen::Matrix2d grad_d_ex = exact_displacement_grad(x, y, t);

            const double ed_x = d_num[0] - d_ex(0);
            const double ed_y = d_num[1] - d_ex(1);
            const double ed_l2_loc = ed_x * ed_x + ed_y * ed_y;

            double ed_grad_sq = 0.0;
            for (int r = 0; r < 2; ++r)
                for (int s = 0; s < 2; ++s) {
                    const double diff = grad_d_num[r][s] - grad_d_ex(r, s);
                    ed_grad_sq += diff * diff;
                }
            d_l2_sq += ed_l2_loc * JxW;
            d_h1_sq += (ed_l2_loc + ed_grad_sq) * JxW;

            // Pressione
            double p_num = 0.0;
            dealii::Tensor<1, 2> grad_p_num;
            for (unsigned int k = 0; k < dofs_p; ++k) {
                const double val_k = p_h(static_cast<Eigen::Index>(loc_p[k]));
                p_num      += val_k * fe_val_p.shape_value(k, q);
                grad_p_num += val_k * fe_val_p.shape_grad(k, q);
            }
            const double ep = p_num - exact_pressure(x, y, t);
            const Eigen::Vector2d gp_ex = exact_pressure_grad(x, y, t);
            const double egp_sq =
                std::pow(grad_p_num[0] - gp_ex(0), 2) +
                std::pow(grad_p_num[1] - gp_ex(1), 2);
            p_l2_sq += (ep * ep) * JxW;
            p_h1_sq += (ep * ep + egp_sq) * JxW;

            // Tracciante
            double c_num = 0.0;
            dealii::Tensor<1, 2> grad_c_num;
            for (unsigned int k = 0; k < dofs_c; ++k) {
                const double val_k = c_h(static_cast<Eigen::Index>(loc_c[k]));
                c_num      += val_k * fe_val_c.shape_value(k, q);
                grad_c_num += val_k * fe_val_c.shape_grad(k, q);
            }
            const double ec = c_num - exact_tracer(x, y, t);
            const Eigen::Vector2d gc_ex = exact_tracer_grad(x, y, t);
            const double egc_sq =
                std::pow(grad_c_num[0] - gc_ex(0), 2) +
                std::pow(grad_c_num[1] - gc_ex(1), 2);
            c_l2_sq += (ec * ec) * JxW;
            c_h1_sq += (ec * ec + egc_sq) * JxW;
        }
    }

    MMSErrorReport rep;
    rep.h_mesh = 1.0 / std::sqrt(static_cast<double>(fem.triangulation().n_active_cells()));
    rep.dt_max = dt_max;
    rep.d_L2   = std::sqrt(d_l2_sq);
    rep.d_H1   = std::sqrt(d_h1_sq);
    rep.p_L2   = std::sqrt(p_l2_sq);
    rep.p_H1   = std::sqrt(p_h1_sq);
    rep.c_L2   = std::sqrt(c_l2_sq);
    rep.c_H1   = std::sqrt(c_h1_sq);
    return rep;
}

MMSErrorReport MMSManufacturedSystem::run_full_mms_test(
    unsigned int subdivisions_N,
    const TimeGrid& grid,
    unsigned int fe_deg_disp,
    unsigned int fe_deg_press,
    unsigned int fe_deg_tracer) const {
    InvariantFEMMatrices2D fem(
        subdivisions_N, physics_, fe_deg_disp, fe_deg_press, fe_deg_tracer, true);

    Eigen::VectorXd d_n = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(fem.num_disp_dofs()));
    Eigen::VectorXd p_n = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(fem.num_press_dofs()));
    Eigen::VectorXd c_n = project_initial_tracer(fem);

    BiotViscoStepSolver biot_solver(fem, theta_, grid);
    TracerTransportStepSolver tracer_solver(fem, theta_, grid, false);
    biot_solver.reset(d_n, p_n);
    tracer_solver.reset(c_n);

    double dt_max = 0.0;
    for (std::size_t n = 1; n <= grid.num_steps(); ++n) {
        const double t_n  = grid.time_at(n);
        dt_max = std::max(dt_max, grid.dt(n));

        const Eigen::VectorXd F_s = assemble_F_mech_rhs(fem, t_n);
        const Eigen::VectorXd S_f = assemble_S_fluid_rhs(fem, t_n);
        const Eigen::VectorXd S_c = assemble_S_tracer_rhs(fem, t_n);

        biot_solver.advance_step(n, F_s, S_f, d_n, p_n);
        tracer_solver.advance_step(n, p_n, S_c, c_n);
    }

    return compute_fem_errors(fem, grid.time_at(grid.num_steps()), d_n, p_n, c_n, dt_max);
}

} // namespace reteco