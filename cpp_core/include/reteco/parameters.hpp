#pragma once
#include <stdexcept>
#include <string>
#include <cmath>

namespace reteco {

struct ParameterTheta {
    double alpha = 0.70; // Ordine frazionario accumulo tracciante in (0, 1)
    double beta  = 0.45; // Ordine frazionario memoria Kelvin-Voigt in (0, 1)
    double D     = 0.08; // Coefficiente diffusivo adimensionale D = D_phys * T0 / L0^2 > 0
    double E     = 2.50; // Modulo elastico adimensionale E = E_phys / E0 > 0

    void validate() const {
        if (!std::isfinite(alpha) || !(alpha > 0.0 && alpha < 1.0)) {
            throw std::invalid_argument("ParameterTheta: alpha must be finite and in (0, 1).");
        }
        if (!std::isfinite(beta) || !(beta > 0.0 && beta < 1.0)) {
            throw std::invalid_argument("ParameterTheta: beta must be finite and in (0, 1).");
        }
        if (!std::isfinite(D) || !(D > 0.0)) {
            throw std::invalid_argument("ParameterTheta: D must be finite and strictly positive.");
        }
        if (!std::isfinite(E) || !(E > 0.0)) {
            throw std::invalid_argument("ParameterTheta: E must be finite and strictly positive.");
        }
    }
};

struct FixedPhysicsConfig {
    double nu    = 0.35; // Coefficiente di Poisson in (-1, 0.5)
    double b     = 0.80; // Coefficiente di Biot in (0, 1]
    double S_p   = 0.50; // Coefficiente di immagazzinamento adimensionale > 0
    double K_h   = 0.20; // Mobilita' idraulica isotropa adimensionale > 0
    double eta_v = 0.45; // Coefficiente viscoso adimensionale > 0
    double tau_v = 1.00; // Scala temporale viscoelastica adimensionale > 0
    double phi_0 = 0.20; // Porosita' di riferimento in (0, 1]
    bool plane_strain = true;

    // Scale caratteristiche dimensionali (con vincolo T_0 == tau_c)
    double L_0   = 1.0e-2;
    double T_0   = 100.0;
    double tau_c = 100.0;
    double E_0   = 1000.0;
    double c_ref = 1.0;

    void validate() const {
        if (!std::isfinite(nu) || !(nu > -1.0 && nu < 0.5)) {
            throw std::invalid_argument("FixedPhysicsConfig: nu must be finite and in (-1, 0.5).");
        }
        if (!std::isfinite(b) || !(b > 0.0 && b <= 1.0)) {
            throw std::invalid_argument("FixedPhysicsConfig: Biot b must be finite and in (0, 1].");
        }
        if (!std::isfinite(S_p) || !(S_p > 0.0)) {
            throw std::invalid_argument("FixedPhysicsConfig: S_p must be finite and > 0.");
        }
        if (!std::isfinite(K_h) || !(K_h > 0.0)) {
            throw std::invalid_argument("FixedPhysicsConfig: K_h must be finite and > 0.");
        }
        if (!std::isfinite(eta_v) || !(eta_v > 0.0)) {
            throw std::invalid_argument("FixedPhysicsConfig: eta_v must be finite and > 0.");
        }
        if (!std::isfinite(tau_v) || !(tau_v > 0.0)) {
            throw std::invalid_argument("FixedPhysicsConfig: tau_v must be finite and > 0.");
        }
        if (!std::isfinite(phi_0) || !(phi_0 > 0.0 && phi_0 <= 1.0)) {
            throw std::invalid_argument("FixedPhysicsConfig: phi_0 must be finite and in (0, 1].");
        }
        if (!std::isfinite(L_0) || !(L_0 > 0.0)) {
            throw std::invalid_argument("FixedPhysicsConfig: scale L_0 must be finite and > 0.");
        }
        if (!std::isfinite(T_0) || !(T_0 > 0.0)) {
            throw std::invalid_argument("FixedPhysicsConfig: scale T_0 must be finite and > 0.");
        }
        if (!std::isfinite(tau_c) || !(tau_c > 0.0)) {
            throw std::invalid_argument("FixedPhysicsConfig: scale tau_c must be finite and > 0.");
        }
        if (!std::isfinite(E_0) || !(E_0 > 0.0)) {
            throw std::invalid_argument("FixedPhysicsConfig: scale E_0 must be finite and > 0.");
        }
        if (!std::isfinite(c_ref) || !(c_ref > 0.0)) {
            throw std::invalid_argument("FixedPhysicsConfig: scale c_ref must be finite and > 0.");
        }
        if (std::abs(T_0 - tau_c) > 1.0e-12 * T_0) {
            throw std::invalid_argument("FixedPhysicsConfig: Mathematical contract requires T_0 == tau_c.");
        }
    }

    double mu_0() const {
        return 1.0 / (2.0 * (1.0 + nu));
    }

    double lambda_0() const {
        if (plane_strain) {
            return nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        } else {
            return nu / (1.0 - nu * nu);
        }
    }
};

} // namespace reteco