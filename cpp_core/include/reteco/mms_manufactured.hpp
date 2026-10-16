#pragma once
#include "reteco/invariant_matrices.hpp"
#include "reteco/parameters.hpp"
#include "reteco/time_grid.hpp"
#include <Eigen/Dense>

namespace reteco {

struct MMSErrorReport {
    double h_mesh = 0.0;
    double dt_max = 0.0;
    double d_L2 = 0.0;
    double d_H1 = 0.0;
    double p_L2 = 0.0;
    double p_H1 = 0.0;
    double c_L2 = 0.0;
    double c_H1 = 0.0;
};

/**
 * Implementazione analitica indipendente della soluzione manufatta (Sezione 7 Contratto Matematico):
 *   d*(x,y,t) = t^2 * ( sin(pi*x)*sin(pi*y), sin(2*pi*x)*sin(pi*y) )^T
 *   p*(x,y,t) = t^2 * cos(pi*x)*cos(pi*y)
 *   c*(x,y,t) = 1 + delta * t^2 * cos(pi*x)*cos(pi*y)
 *   q*(x,y,t) = -K_h * grad(p*)
 *
 * Le forzanti f_s*, s_f*, s_c* sono valutate nei punti di quadratura di Gauss direttamente
 * dalle derivate analitiche continue, senza mai usare le matrici del solutore.
 */
class MMSManufacturedSystem {
public:
    MMSManufacturedSystem(
        const FixedPhysicsConfig& physics,
        const ParameterTheta& theta,
        double delta_tracer = 0.4);

    // Valori puntuali analitici
    Eigen::Vector2d exact_displacement(double x, double y, double t) const;
    Eigen::Matrix2d exact_displacement_grad(double x, double y, double t) const;

    double exact_pressure(double x, double y, double t) const;
    Eigen::Vector2d exact_pressure_grad(double x, double y, double t) const;

    double exact_tracer(double x, double y, double t) const;
    Eigen::Vector2d exact_tracer_grad(double x, double y, double t) const;

    // Forzanti continue esatte
    Eigen::Vector2d exact_f_s(double x, double y, double t) const;
    double exact_s_f(double x, double y, double t) const;
    double exact_s_c(double x, double y, double t) const;

    // Assemblaggio dei termini noti FEM mediante quadratura di Gauss di ordine elevato
    Eigen::VectorXd assemble_F_mech_rhs(const InvariantFEMMatrices2D& fem, double t) const;
    Eigen::VectorXd assemble_S_fluid_rhs(const InvariantFEMMatrices2D& fem, double t) const;
    Eigen::VectorXd assemble_S_tracer_rhs(const InvariantFEMMatrices2D& fem, double t) const;

    // Interpolazione del dato iniziale c*(x,y,0) = 1.0
    Eigen::VectorXd project_initial_tracer(const InvariantFEMMatrices2D& fem) const;

    // Calcolo esatto degli errori L^2(Omega) e H^1(Omega) con quadratura di Gauss a 5 punti per direzione
    MMSErrorReport compute_fem_errors(
        const InvariantFEMMatrices2D& fem,
        double t,
        const Eigen::VectorXd& d_h,
        const Eigen::VectorXd& p_h,
        const Eigen::VectorXd& c_h,
        double dt_max = 0.0) const;

    // Esegue una simulazione MMS completa da t=0 a t=T e restituisce gli errori al tempo finale T
    MMSErrorReport run_full_mms_test(
        unsigned int subdivisions_N,
        const TimeGrid& grid,
        unsigned int fe_deg_disp = 2,
        unsigned int fe_deg_press = 1,
        unsigned int fe_deg_tracer = 1) const;

private:
    FixedPhysicsConfig physics_;
    ParameterTheta theta_;
    double delta_;
    double caputo_coeff_beta_;  // Gamma(3) / Gamma(3 - beta)
    double caputo_coeff_alpha_; // Gamma(3) / Gamma(3 - alpha)
};

} // namespace reteco