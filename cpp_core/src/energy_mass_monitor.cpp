#include "reteco/energy_mass_monitor.hpp"
#include "reteco/caputo_l1_memory.hpp"
#include <cmath>
#include <stdexcept>

namespace reteco {

EnergyMassMonitor::EnergyMassMonitor(const InvariantFEMMatrices2D& fem)
    : fem_(fem),
      ones_c_(Eigen::VectorXd::Ones(static_cast<Eigen::Index>(fem.num_tracer_dofs()))) {}

double EnergyMassMonitor::compute_tracer_mass(const Eigen::VectorXd& c_dofs) const {
    if (static_cast<std::size_t>(c_dofs.size()) != fem_.num_tracer_dofs()) {
        throw std::invalid_argument("EnergyMassMonitor::compute_tracer_mass: c_dofs size mismatch.");
    }
    return ones_c_.dot(fem_.M_c() * c_dofs);
}

double EnergyMassMonitor::compute_mechanical_fluid_energy(
    double E_modulus,
    const Eigen::VectorXd& d_dofs,
    const Eigen::VectorXd& p_dofs) const {
    if (static_cast<std::size_t>(d_dofs.size()) != fem_.num_disp_dofs() ||
        static_cast<std::size_t>(p_dofs.size()) != fem_.num_press_dofs()) {
        throw std::invalid_argument("EnergyMassMonitor::compute_mechanical_fluid_energy: size mismatch.");
    }
    const double el_energy = 0.5 * E_modulus * d_dofs.dot(fem_.K0_mech() * d_dofs);
    const double fl_energy = 0.5 * fem_.physics().S_p * p_dofs.dot(fem_.M_p() * p_dofs);
    return el_energy + fl_energy;
}

std::vector<StepDiagnostics> EnergyMassMonitor::evaluate_trajectory_diagnostics(
    const ParameterTheta& theta,
    const TimeGrid& grid,
    const std::vector<Eigen::VectorXd>& d_hist,
    const std::vector<Eigen::VectorXd>& p_hist,
    const std::vector<Eigen::VectorXd>& c_hist,
    const std::vector<Eigen::VectorXd>& F_mech_hist,
    const std::vector<Eigen::VectorXd>& S_fluid_hist) const {
    theta.validate();
    const std::size_t N = grid.num_steps();
    if (d_hist.size() != N + 1 || p_hist.size() != N + 1 || c_hist.size() != N + 1 ||
        F_mech_hist.size() != N + 1 || S_fluid_hist.size() != N + 1) {
        throw std::invalid_argument(
            "EnergyMassMonitor::evaluate_trajectory_diagnostics: trajectory length must be num_steps + 1.");
    }

    const auto& phys      = fem_.physics();
    const double visco_sc = phys.eta_v * std::pow(phys.tau_v, theta.beta);
    const auto& is_dir    = fem_.is_dirichlet_disp();

    std::vector<StepDiagnostics> out(N + 1);
    CaputoL1Memory mem_beta(theta.beta, grid, fem_.num_disp_dofs());
    mem_beta.reset(d_hist[0]);

    const double E_init = compute_mechanical_fluid_energy(theta.E, d_hist[0], p_hist[0]);

    out[0].time                 = grid.time_at(0);
    out[0].elastic_energy       = 0.5 * theta.E * d_hist[0].dot(fem_.K0_mech() * d_hist[0]);
    out[0].fluid_storage_energy = 0.5 * phys.S_p * p_hist[0].dot(fem_.M_p() * p_hist[0]);
    out[0].total_energy         = E_init;
    out[0].tracer_total_mass    = compute_tracer_mass(c_hist[0]);

    double cum_visco = 0.0;
    double cum_darcy = 0.0;
    double cum_work  = 0.0;
    double prev_R    = 0.0;

    for (std::size_t n = 1; n <= N; ++n) {
        if (static_cast<std::size_t>(d_hist[n].size()) != fem_.num_disp_dofs() ||
            static_cast<std::size_t>(p_hist[n].size()) != fem_.num_press_dofs() ||
            static_cast<std::size_t>(c_hist[n].size()) != fem_.num_tracer_dofs() ||
            static_cast<std::size_t>(F_mech_hist[n].size()) != fem_.num_disp_dofs() ||
            static_cast<std::size_t>(S_fluid_hist[n].size()) != fem_.num_press_dofs()) {
            throw std::invalid_argument("EnergyMassMonitor: step vector size mismatch.");
        }

        const double dt_n = grid.dt(n);
        const Eigen::VectorXd delta_d = d_hist[n] - d_hist[n - 1];
        const Eigen::VectorXd delta_p = p_hist[n] - p_hist[n - 1];
        const Eigen::VectorXd v_n     = delta_d / dt_n;

        const Eigen::VectorXd caputo_d_n = mem_beta.evaluate_discrete_derivative(n, d_hist[n]);
        mem_beta.commit_step(n, d_hist[n]);

        // Azzeriamo i DOFs di Dirichlet nella forzante meccanica esattamente come fa il solutore
        Eigen::VectorXd F_eff = F_mech_hist[n];
        for (std::size_t i = 0; i < fem_.num_disp_dofs(); ++i) {
            if (is_dir[i]) F_eff(static_cast<Eigen::Index>(i)) = 0.0;
        }

        const double step_visco = dt_n * visco_sc * caputo_d_n.dot(fem_.K0_mech() * v_n);
        const double step_darcy = dt_n * phys.K_h * p_hist[n].dot(fem_.K_p() * p_hist[n]);
        const double step_work  = dt_n * (F_eff.dot(v_n) + S_fluid_hist[n].dot(p_hist[n]));

        cum_visco += step_visco;
        cum_darcy += step_darcy;
        cum_work  += step_work;

        StepDiagnostics diag;
        diag.time                     = grid.time_at(n);
        diag.elastic_energy           = 0.5 * theta.E * d_hist[n].dot(fem_.K0_mech() * d_hist[n]);
        diag.fluid_storage_energy     = 0.5 * phys.S_p * p_hist[n].dot(fem_.M_p() * p_hist[n]);
        diag.total_energy             = diag.elastic_energy + diag.fluid_storage_energy;
        diag.cumulative_visco_dissip  = cum_visco;
        diag.cumulative_darcy_dissip  = cum_darcy;
        diag.cumulative_external_work = cum_work;
        diag.energy_balance_residual  =
            (diag.total_energy + cum_visco + cum_darcy) - (E_init + cum_work);

        // Dissipazione numerica di passo (Sez. 2.1 di ChatGPT):
        // D_num,n = 0.5 * E * ||Delta d^n||_{K_0}^2 + 0.5 * S_p * ||Delta p^n||_{M_p}^2
        diag.step_numerical_dissipation =
            0.5 * theta.E * delta_d.dot(fem_.K0_mech() * delta_d) +
            0.5 * phys.S_p * delta_p.dot(fem_.M_p() * delta_p);

        const double delta_R = diag.energy_balance_residual - prev_R;
        diag.step_discrete_identity_error = std::abs(delta_R + diag.step_numerical_dissipation);
        prev_R = diag.energy_balance_residual;

        diag.tracer_total_mass = compute_tracer_mass(c_hist[n]);
        out[n] = diag;
    }

    return out;
}

} // namespace reteco