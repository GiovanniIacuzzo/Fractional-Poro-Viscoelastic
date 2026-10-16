#pragma once
#include "reteco/invariant_matrices.hpp"
#include "reteco/parameters.hpp"
#include <Eigen/Dense>

namespace reteco {

/**
 * Verifica e risolve la condizione di compatibilita' meccanica iniziale (Eq. 14 / Sez. 4 Contratto):
 *   a_E(d_0, v) - b * (p_0, div(v)) = <F(0), v>   per ogni v in V.
 */
class InitialCompatibility {
public:
    // Calcola lo spostamento iniziale compatibile d_0 dato p_0 e il vettore di carico meccanico F_0
    static Eigen::VectorXd compute_compatible_d0(
        const InvariantFEMMatrices2D& fem,
        double E_modulus,
        const Eigen::VectorXd& p0,
        const Eigen::VectorXd& F0);

    // Restituisce la norma L2 discreta del residuo di compatibilita' iniziale sulle componenti libere in V
    static double compatibility_residual_norm(
        const InvariantFEMMatrices2D& fem,
        double E_modulus,
        const Eigen::VectorXd& d0,
        const Eigen::VectorXd& p0,
        const Eigen::VectorXd& F0);
};

} // namespace reteco