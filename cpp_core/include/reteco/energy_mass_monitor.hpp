#pragma once
#include "reteco/invariant_matrices.hpp"
#include "reteco/parameters.hpp"
#include "reteco/time_grid.hpp"
#include <Eigen/Dense>
#include <vector>

namespace reteco {

struct StepDiagnostics {
    double time = 0.0;
    double elastic_energy = 0.0;              // 0.5 * a_E(d^n, d^n)
    double fluid_storage_energy = 0.0;        // 0.5 * S_p * ||p^n||_{L^2}^2
    double total_energy = 0.0;                // E(t_n)
    double cumulative_visco_dissip = 0.0;     // D_beta(t_n)
    double cumulative_darcy_dissip = 0.0;     // sum_{m=1}^n dt_m * a_h(p^m, p^m)
    double cumulative_external_work = 0.0;    // sum_{m=1}^n dt_m * (<F^m, v^m> + <s_f^m, p^m>)
    double energy_balance_residual = 0.0;     // R_n = E(t_n) + D_beta(t_n) + D_h(t_n) - E(0) - Work
    double step_numerical_dissipation = 0.0;  // 0.5*E*||Delta d^n||_{K0}^2 + 0.5*S_p*||Delta p^n||_{Mp}^2
    double step_discrete_identity_error = 0.0;// |(R_n - R_{n-1}) + step_numerical_dissipation|
    double tracer_total_mass = 0.0;           // M(t_n) = int_Omega c_h(x, t_n) dx
    double tracer_storage_mass = 0.0;        // M_phi(t_n) = int_Omega phi(x) c_h(x,t_n) dx
};

class EnergyMassMonitor {
public:
    explicit EnergyMassMonitor(const InvariantFEMMatrices2D& fem);

    double compute_tracer_mass(const Eigen::VectorXd& c_dofs) const;
    // Storage mass with piecewise phi(x), matching the discretized tracer PDE.
    double compute_tracer_storage_mass(const Eigen::VectorXd& c_dofs) const;

    double compute_mechanical_fluid_energy(
        double E_modulus,
        const Eigen::VectorXd& d_dofs,
        const Eigen::VectorXd& p_dofs) const;

    std::vector<StepDiagnostics> evaluate_trajectory_diagnostics(
        const ParameterTheta& theta,
        const TimeGrid& grid,
        const std::vector<Eigen::VectorXd>& d_hist,
        const std::vector<Eigen::VectorXd>& p_hist,
        const std::vector<Eigen::VectorXd>& c_hist,
        const std::vector<Eigen::VectorXd>& F_mech_hist,
        const std::vector<Eigen::VectorXd>& S_fluid_hist) const;

private:
    const InvariantFEMMatrices2D& fem_;
    Eigen::VectorXd ones_c_;
};

} // namespace reteco