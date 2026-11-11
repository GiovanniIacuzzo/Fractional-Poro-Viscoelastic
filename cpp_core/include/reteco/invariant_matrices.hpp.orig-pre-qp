#pragma once
#include "reteco/parameters.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <string>
#include <vector>

namespace reteco {

// Multipliers relative to homogeneous reference coefficients. Material ID 0
// denotes the unsegmented background; 1/2/4 are BraTS labels (NCR/ED/ET).
// Phenomenological sensitivity parameters, not patient-calibrated properties.
struct RegionMaterialMultipliers {
    double stiffness = 1.0;
    double hydraulic_mobility = 1.0;
    double tracer_diffusivity = 1.0;
    double porosity = 1.0;

    void validate(double phi0) const;
};

class InvariantFEMMatrices2D {
public:
    // 1. Costruttore predefinito sul quadrato unitario [0, 1]^2 (compatibilita' MMS completa)
    InvariantFEMMatrices2D(
        unsigned int subdivisions_N,
        const FixedPhysicsConfig& physics,
        unsigned int fe_deg_disp = 2,
        unsigned int fe_deg_press = 1,
        unsigned int fe_deg_tracer = 1,
        bool dirichlet_all_displacement_boundary = true);

    // 2. Costruttore da file di mesh quadrilatera verificata (.msh o .inp)
    InvariantFEMMatrices2D(
        const std::string& mesh_file_path,
        const FixedPhysicsConfig& physics,
        unsigned int fe_deg_disp = 2,
        unsigned int fe_deg_press = 1,
        unsigned int fe_deg_tracer = 1,
        const std::vector<unsigned int>& dirichlet_disp_boundary_ids = {1},
        const std::map<unsigned int, RegionMaterialMultipliers>& region_materials = {});

    ~InvariantFEMMatrices2D() {
        dh_disp_.clear();
        dh_press_.clear();
        dh_tracer_.clear();
    }

    std::size_t num_disp_dofs() const { return dh_disp_.n_dofs(); }
    std::size_t num_press_dofs() const { return dh_press_.n_dofs(); }
    std::size_t num_tracer_dofs() const { return dh_tracer_.n_dofs(); }
    std::size_t num_active_cells() const { return tria_.n_active_cells(); }
    std::size_t num_vertices() const { return tria_.n_used_vertices(); }

    const FixedPhysicsConfig& physics() const { return physics_; }
    bool is_imported_mesh() const { return is_imported_mesh_; }
    const std::string& mesh_source_description() const { return mesh_source_description_; }

    const Eigen::SparseMatrix<double>& M_d() const { return M_d_; }
    // Elastic matrix includes material-dependent stiffness multipliers.
    const Eigen::SparseMatrix<double>& K0_mech() const { return K0_mech_; }
    // Viscoelastic kernel remains homogeneous: eta_v is not changed by E(x).
    const Eigen::SparseMatrix<double>& Kvisco_mech() const { return Kvisco_mech_; }
    const Eigen::SparseMatrix<double>& B_biot() const { return B_biot_; }
    const Eigen::SparseMatrix<double>& M_p() const { return M_p_; }
    const Eigen::SparseMatrix<double>& K_p() const { return K_p_; }
    const Eigen::SparseMatrix<double>& M_c() const { return M_c_; }
    const Eigen::SparseMatrix<double>& K_c() const { return K_c_; }
    // Normalized by phi_0 and D; these matrices enter the heterogeneous
    // tracer storage/diffusion operator (M_phi_c and K_phiD_c).
    const Eigen::SparseMatrix<double>& M_phi_c() const { return M_phi_c_; }
    const Eigen::SparseMatrix<double>& K_phiD_c() const { return K_phiD_c_; }
    const std::map<unsigned int, RegionMaterialMultipliers>& region_materials() const { return region_materials_; }

    double compute_disp_L2_norm(const Eigen::VectorXd& d_dofs) const;
    double compute_press_L2_norm(const Eigen::VectorXd& p_dofs) const;
    double compute_tracer_L2_norm(const Eigen::VectorXd& c_dofs) const;

    const std::vector<bool>& is_dirichlet_disp() const { return is_dirichlet_disp_; }
    const std::vector<unsigned int>& get_boundary_ids() const { return discovered_boundary_ids_; }
    const std::vector<unsigned int>& get_dirichlet_disp_boundary_ids() const { return dirichlet_disp_boundary_ids_; }
    const std::map<unsigned int, unsigned int>& get_boundary_face_counts() const { return boundary_face_counts_; }

    Eigen::SparseMatrix<double> assemble_tracer_advection(
        const Eigen::VectorXd& p_dofs,
        bool supg_stabilization = false,
        double D_tracer = 0.0) const;

    double compute_max_darcy_velocity(const Eigen::VectorXd& p_dofs) const;

    Eigen::MatrixXd get_cell_centers() const;
    Eigen::MatrixXd compute_cell_darcy_fluxes(const Eigen::VectorXd& p_dofs) const;

    Eigen::MatrixXd get_disp_support_points() const;
    Eigen::MatrixXd get_press_support_points() const;
    Eigen::MatrixXd get_tracer_support_points() const;

    Eigen::MatrixXd get_mesh_vertices() const;
    Eigen::MatrixXi get_cell_quad_connectivity() const;
    Eigen::MatrixXi get_cell_tracer_dof_connectivity() const;
    Eigen::VectorXi get_cell_material_ids() const;

    bool verify_dof_and_mesh_consistency(double tol = 1.0e-11) const;

    const dealii::Triangulation<2>& triangulation() const { return tria_; }
    const dealii::DoFHandler<2>& dof_handler_disp() const { return dh_disp_; }
    const dealii::DoFHandler<2>& dof_handler_press() const { return dh_press_; }
    const dealii::DoFHandler<2>& dof_handler_tracer() const { return dh_tracer_; }

private:
    void setup_dofs_and_assemble_all();
    const RegionMaterialMultipliers& material(unsigned int id) const;

    FixedPhysicsConfig physics_;
    std::map<unsigned int, RegionMaterialMultipliers> region_materials_;
    bool is_imported_mesh_ = false;
    std::string mesh_source_description_;

    dealii::Triangulation<2> tria_;
    dealii::FESystem<2> fe_disp_;
    dealii::FE_Q<2> fe_press_;
    dealii::FE_Q<2> fe_tracer_;

    dealii::DoFHandler<2> dh_disp_;
    dealii::DoFHandler<2> dh_press_;
    dealii::DoFHandler<2> dh_tracer_;

    std::vector<unsigned int> discovered_boundary_ids_;
    std::vector<unsigned int> dirichlet_disp_boundary_ids_;
    std::map<unsigned int, unsigned int> boundary_face_counts_;
    std::vector<bool> is_dirichlet_disp_;

    Eigen::SparseMatrix<double> M_d_;
    Eigen::SparseMatrix<double> K0_mech_;
    Eigen::SparseMatrix<double> Kvisco_mech_;
    Eigen::SparseMatrix<double> B_biot_;
    Eigen::SparseMatrix<double> M_p_;
    Eigen::SparseMatrix<double> K_p_;
    Eigen::SparseMatrix<double> M_c_;
    Eigen::SparseMatrix<double> K_c_;
    Eigen::SparseMatrix<double> M_phi_c_;
    Eigen::SparseMatrix<double> K_phiD_c_;
};

} // namespace reteco