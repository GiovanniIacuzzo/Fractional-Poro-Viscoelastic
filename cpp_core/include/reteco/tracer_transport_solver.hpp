#pragma once
#include "reteco/invariant_matrices.hpp"
#include "reteco/caputo_l1_memory.hpp"
#include "reteco/parameters.hpp"

namespace reteco {

/**
 * Risolve al passo t_n l'equazione debole del tracciante passivo (Eq. 5 Contratto Matematico):
 *
 *   phi_0 * (CD_t^alpha c^n, z) - (c^n * q^n, grad(z)) + phi_0 * D * (grad(c^n), grad(z)) = (s_c^n, z)
 */
class TracerTransportStepSolver {
public:
    TracerTransportStepSolver(
        const InvariantFEMMatrices2D& fem,
        const ParameterTheta& theta,
        const TimeGrid& grid,
        bool use_supg = false);

    void reset(const Eigen::VectorXd& c0);

    void advance_step(
        std::size_t n,
        const Eigen::VectorXd& p_n,
        const Eigen::VectorXd& S_tracer_n,
        Eigen::VectorXd& c_n_out);

private:
    const InvariantFEMMatrices2D& fem_;
    ParameterTheta theta_;
    TimeGrid grid_;
    CaputoL1Memory mem_alpha_;
    bool use_supg_;
};

} // namespace reteco