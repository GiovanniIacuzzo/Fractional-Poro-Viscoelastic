#include "reteco/invariant_matrices.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

namespace reteco {

void RegionMaterialMultipliers::validate(double phi0) const {
    for (const double v : {stiffness, hydraulic_mobility, tracer_diffusivity, porosity}) {
        if (!std::isfinite(v) || v <= 0.0) {
            throw std::invalid_argument("Regional material multipliers must be finite and > 0.");
        }
    }
    if (phi0 * porosity > 1.0) {
        throw std::invalid_argument("Regional porosity phi_0*multiplier must not exceed 1.");
    }
}

const RegionMaterialMultipliers& InvariantFEMMatrices2D::material(unsigned int id) const {
    static const RegionMaterialMultipliers unit{};
    if (region_materials_.empty()) return unit;
    const auto it = region_materials_.find(id);
    if (it == region_materials_.end()) {
        throw std::invalid_argument("Missing regional material for cell material_id=" + std::to_string(id));
    }
    return it->second;
}

InvariantFEMMatrices2D::InvariantFEMMatrices2D(
    unsigned int subdivisions_N,
    const FixedPhysicsConfig& physics,
    unsigned int fe_deg_disp,
    unsigned int fe_deg_press,
    unsigned int fe_deg_tracer,
    bool dirichlet_all_displacement_boundary)
    : physics_(physics),
      is_imported_mesh_(false),
      mesh_source_description_("unit_square_subdivided_" + std::to_string(subdivisions_N)),
      fe_disp_(dealii::FE_Q<2>(fe_deg_disp), 2),
      fe_press_(fe_deg_press),
      fe_tracer_(fe_deg_tracer),
      dh_disp_(tria_),
      dh_press_(tria_),
      dh_tracer_(tria_) {
    if (subdivisions_N < 2) {
        throw std::invalid_argument("subdivisions_N deve essere >= 2.");
    }
    physics_.validate();

    dealii::GridGenerator::subdivided_hyper_cube(tria_, subdivisions_N, 0.0, 1.0, true);

    const auto b_ids = tria_.get_boundary_ids();
    for (const auto bid : b_ids) {
        discovered_boundary_ids_.push_back(static_cast<unsigned int>(bid));
    }
    std::sort(discovered_boundary_ids_.begin(), discovered_boundary_ids_.end());

    if (dirichlet_all_displacement_boundary) {
        dirichlet_disp_boundary_ids_ = discovered_boundary_ids_;
    } else {
        dirichlet_disp_boundary_ids_ = {0u};
    }

    setup_dofs_and_assemble_all();
}

InvariantFEMMatrices2D::InvariantFEMMatrices2D(
    const std::string& mesh_file_path,
    const FixedPhysicsConfig& physics,
    unsigned int fe_deg_disp,
    unsigned int fe_deg_press,
    unsigned int fe_deg_tracer,
    const std::vector<unsigned int>& dirichlet_disp_boundary_ids,
    const std::map<unsigned int, RegionMaterialMultipliers>& region_materials)
    : physics_(physics),
      region_materials_(region_materials),
      is_imported_mesh_(true),
      mesh_source_description_(mesh_file_path),
      fe_disp_(dealii::FE_Q<2>(fe_deg_disp), 2),
      fe_press_(fe_deg_press),
      fe_tracer_(fe_deg_tracer),
      dh_disp_(tria_),
      dh_press_(tria_),
      dh_tracer_(tria_),
      dirichlet_disp_boundary_ids_(dirichlet_disp_boundary_ids) {
    physics_.validate();
    for (const auto& [id, coeff] : region_materials_) {
        (void) id;
        coeff.validate(physics_.phi_0);
    }

    std::filesystem::path p(mesh_file_path);
    if (!std::filesystem::exists(p)) {
        throw std::runtime_error("File di mesh anatomica non trovato: " + mesh_file_path);
    }

    std::ifstream in(mesh_file_path);
    if (!in.is_open()) {
        throw std::runtime_error("Impossibile aprire il file di mesh: " + mesh_file_path);
    }

    dealii::GridIn<2> grid_in;
    grid_in.attach_triangulation(tria_);
    const std::string ext = p.extension().string();
    if (ext == ".msh") {
        grid_in.read_msh(in);
    } else if (ext == ".inp" || ext == ".ucd") {
        grid_in.read_ucd(in);
    } else {
        throw std::invalid_argument(
            "Formato mesh non supportato ('" + ext + "'). Usa .msh (Gmsh v2.2) o .inp (AVS UCD).");
    }

    if (tria_.n_active_cells() == 0) {
        throw std::runtime_error("La mesh importata non contiene celle attive: " + mesh_file_path);
    }

    const auto b_ids = tria_.get_boundary_ids();
    for (const auto bid : b_ids) {
        discovered_boundary_ids_.push_back(static_cast<unsigned int>(bid));
    }
    std::sort(discovered_boundary_ids_.begin(), discovered_boundary_ids_.end());

    if (dirichlet_disp_boundary_ids_.empty()) {
        throw std::invalid_argument(
            "dirichlet_disp_boundary_ids non puo' essere vuoto per il problema elastostatico/Biot.");
    }

    for (const unsigned int req_bid : dirichlet_disp_boundary_ids_) {
        if (std::find(
                discovered_boundary_ids_.begin(),
                discovered_boundary_ids_.end(),
                req_bid) == discovered_boundary_ids_.end()) {
            throw std::invalid_argument(
                "Il boundary_id di Dirichlet richiesto (" + std::to_string(req_bid) +
                ") non esiste tra i boundary_id effettivi della mesh importata.");
        }
    }

    setup_dofs_and_assemble_all();
}

void InvariantFEMMatrices2D::setup_dofs_and_assemble_all() {
    boundary_face_counts_.clear();
    for (const auto& cell : tria_.active_cell_iterators()) {
        if (cell->measure() <= 0.0 || !std::isfinite(cell->measure())) {
            throw std::runtime_error("Rilevata cella quadrilatera con misura <= 0 nella Triangulation<2>.");
        }
        for (const auto& face : cell->face_iterators()) {
            if (face->at_boundary()) {
                const unsigned int bid = static_cast<unsigned int>(face->boundary_id());
                boundary_face_counts_[bid] += 1u;
            }
        }
    }

    dh_disp_.distribute_dofs(fe_disp_);
    dh_press_.distribute_dofs(fe_press_);
    dh_tracer_.distribute_dofs(fe_tracer_);

    const std::size_t n_d = dh_disp_.n_dofs();
    const std::size_t n_p = dh_press_.n_dofs();
    const std::size_t n_c = dh_tracer_.n_dofs();

    is_dirichlet_disp_.assign(n_d, false);
    std::map<dealii::types::global_dof_index, double> bval;
    dealii::Functions::ZeroFunction<2> zero_func(2);
    for (const unsigned int bid : dirichlet_disp_boundary_ids_) {
        dealii::VectorTools::interpolate_boundary_values(
            dh_disp_,
            static_cast<dealii::types::boundary_id>(bid),
            zero_func,
            bval);
    }
    for (const auto& kv : bval) {
        is_dirichlet_disp_[kv.first] = true;
    }

    const unsigned int quad_deg = std::max({fe_disp_.degree, fe_press_.degree, fe_tracer_.degree}) + 2;
    dealii::QGauss<2> quadrature(quad_deg);

    dealii::FEValues<2> fe_val_d(
        fe_disp_, quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    dealii::FEValues<2> fe_val_p(
        fe_press_, quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    dealii::FEValues<2> fe_val_c(
        fe_tracer_, quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int dofs_per_cell_d = fe_disp_.n_dofs_per_cell();
    const unsigned int dofs_per_cell_p = fe_press_.n_dofs_per_cell();
    const unsigned int dofs_per_cell_c = fe_tracer_.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature.size();

    const double mu0 = physics_.mu_0();
    const double lam0 = physics_.lambda_0();

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> trip_Md, trip_K0, trip_Kvisco, trip_B, trip_Mp, trip_Kp, trip_Mc, trip_Kc;
    std::vector<Triplet> trip_Mphi, trip_KphiD;
    trip_Md.reserve(tria_.n_active_cells() * dofs_per_cell_d * dofs_per_cell_d);
    trip_K0.reserve(tria_.n_active_cells() * dofs_per_cell_d * dofs_per_cell_d);
    trip_Kvisco.reserve(trip_K0.capacity());
    trip_B.reserve(tria_.n_active_cells() * dofs_per_cell_d * dofs_per_cell_p);
    trip_Mp.reserve(tria_.n_active_cells() * dofs_per_cell_p * dofs_per_cell_p);
    trip_Kp.reserve(tria_.n_active_cells() * dofs_per_cell_p * dofs_per_cell_p);
    trip_Mc.reserve(tria_.n_active_cells() * dofs_per_cell_c * dofs_per_cell_c);
    trip_Kc.reserve(tria_.n_active_cells() * dofs_per_cell_c * dofs_per_cell_c);
    trip_Mphi.reserve(trip_Mc.capacity());
    trip_KphiD.reserve(trip_Kc.capacity());

    std::vector<dealii::types::global_dof_index> loc_d(dofs_per_cell_d);
    std::vector<dealii::types::global_dof_index> loc_p(dofs_per_cell_p);
    std::vector<dealii::types::global_dof_index> loc_c(dofs_per_cell_c);

    auto cell_d = dh_disp_.begin_active();
    auto cell_p = dh_press_.begin_active();
    auto cell_c = dh_tracer_.begin_active();

    for (; cell_d != dh_disp_.end(); ++cell_d, ++cell_p, ++cell_c) {
        fe_val_d.reinit(cell_d);
        fe_val_p.reinit(cell_p);
        fe_val_c.reinit(cell_c);

        cell_d->get_dof_indices(loc_d);
        cell_p->get_dof_indices(loc_p);
        cell_c->get_dof_indices(loc_c);
        const auto& mat = material(static_cast<unsigned int>(cell_d->material_id()));
        if (cell_d->material_id() != cell_p->material_id() ||
            cell_d->material_id() != cell_c->material_id()) {
            throw std::runtime_error("Cell material IDs differ between DoF handlers.");
        }

        Eigen::MatrixXd loc_Md = Eigen::MatrixXd::Zero(dofs_per_cell_d, dofs_per_cell_d);
        Eigen::MatrixXd loc_K0 = Eigen::MatrixXd::Zero(dofs_per_cell_d, dofs_per_cell_d);
        Eigen::MatrixXd loc_Kvisco = Eigen::MatrixXd::Zero(dofs_per_cell_d, dofs_per_cell_d);
        Eigen::MatrixXd loc_B_mat = Eigen::MatrixXd::Zero(dofs_per_cell_d, dofs_per_cell_p);
        Eigen::MatrixXd loc_Mp_mat = Eigen::MatrixXd::Zero(dofs_per_cell_p, dofs_per_cell_p);
        Eigen::MatrixXd loc_Kp_mat = Eigen::MatrixXd::Zero(dofs_per_cell_p, dofs_per_cell_p);
        Eigen::MatrixXd loc_Mc_mat = Eigen::MatrixXd::Zero(dofs_per_cell_c, dofs_per_cell_c);
        Eigen::MatrixXd loc_Kc_mat = Eigen::MatrixXd::Zero(dofs_per_cell_c, dofs_per_cell_c);
        Eigen::MatrixXd loc_Mphi_mat = Eigen::MatrixXd::Zero(dofs_per_cell_c, dofs_per_cell_c);
        Eigen::MatrixXd loc_KphiD_mat = Eigen::MatrixXd::Zero(dofs_per_cell_c, dofs_per_cell_c);

        for (unsigned int q = 0; q < n_q_points; ++q) {
            const double jxw = fe_val_d.JxW(q);

            std::vector<Eigen::Vector3d> eps_d(dofs_per_cell_d);
            std::vector<double> div_d(dofs_per_cell_d, 0.0);
            std::vector<unsigned int> comp_d(dofs_per_cell_d, 0u);
            std::vector<double> val_d(dofs_per_cell_d, 0.0);

            for (unsigned int i = 0; i < dofs_per_cell_d; ++i) {
                const unsigned int comp_i = fe_disp_.system_to_component_index(i).first;
                comp_d[i] = comp_i;
                val_d[i] = fe_val_d.shape_value(i, q);
                const auto grad_i = fe_val_d.shape_grad(i, q);
                if (comp_i == 0) {
                    eps_d[i] = Eigen::Vector3d(grad_i[0], 0.0, grad_i[1]);
                    div_d[i] = grad_i[0];
                } else {
                    eps_d[i] = Eigen::Vector3d(0.0, grad_i[1], grad_i[0]);
                    div_d[i] = grad_i[1];
                }
            }

            for (unsigned int i = 0; i < dofs_per_cell_d; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell_d; ++j) {
                    if (comp_d[i] == comp_d[j]) {
                        loc_Md(i, j) += val_d[i] * val_d[j] * jxw;
                    }
                    const double double_dot =
                        eps_d[i](0) * eps_d[j](0) +
                        eps_d[i](1) * eps_d[j](1) +
                        0.5 * eps_d[i](2) * eps_d[j](2);
                    const double val = (2.0 * mu0 * double_dot + lam0 * div_d[i] * div_d[j]) * jxw;
                    loc_K0(i, j) += mat.stiffness * val;
                    loc_Kvisco(i, j) += val;
                }
                for (unsigned int m = 0; m < dofs_per_cell_p; ++m) {
                    const double phi_p_m = fe_val_p.shape_value(m, q);
                    loc_B_mat(i, m) += div_d[i] * phi_p_m * jxw;
                }
            }

            for (unsigned int i = 0; i < dofs_per_cell_p; ++i) {
                const double phi_i = fe_val_p.shape_value(i, q);
                const auto grad_i = fe_val_p.shape_grad(i, q);
                for (unsigned int j = 0; j < dofs_per_cell_p; ++j) {
                    const double phi_j = fe_val_p.shape_value(j, q);
                    const auto grad_j = fe_val_p.shape_grad(j, q);
                    loc_Mp_mat(i, j) += phi_i * phi_j * jxw;
                    loc_Kp_mat(i, j) += mat.hydraulic_mobility * (grad_i * grad_j) * jxw;
                }
            }

            for (unsigned int i = 0; i < dofs_per_cell_c; ++i) {
                const double phi_i = fe_val_c.shape_value(i, q);
                const auto grad_i = fe_val_c.shape_grad(i, q);
                for (unsigned int j = 0; j < dofs_per_cell_c; ++j) {
                    const double phi_j = fe_val_c.shape_value(j, q);
                    const auto grad_j = fe_val_c.shape_grad(j, q);
                    const double m = phi_i * phi_j * jxw;
                    const double k = (grad_i * grad_j) * jxw;
                    loc_Mc_mat(i, j) += m;
                    loc_Kc_mat(i, j) += k;
                    loc_Mphi_mat(i, j) += mat.porosity * m;
                    loc_KphiD_mat(i, j) += mat.porosity * mat.tracer_diffusivity * k;
                }
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell_d; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell_d; ++j) {
                if (std::abs(loc_Md(i, j)) > 0.0) {
                    trip_Md.emplace_back(loc_d[i], loc_d[j], loc_Md(i, j));
                }
                if (std::abs(loc_K0(i, j)) > 0.0) {
                    trip_K0.emplace_back(loc_d[i], loc_d[j], loc_K0(i, j));
                }
                if (std::abs(loc_Kvisco(i, j)) > 0.0) {
                    trip_Kvisco.emplace_back(loc_d[i], loc_d[j], loc_Kvisco(i, j));
                }
            }
            for (unsigned int m = 0; m < dofs_per_cell_p; ++m) {
                if (std::abs(loc_B_mat(i, m)) > 0.0) {
                    trip_B.emplace_back(loc_d[i], loc_p[m], loc_B_mat(i, m));
                }
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell_p; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell_p; ++j) {
                if (std::abs(loc_Mp_mat(i, j)) > 0.0) {
                    trip_Mp.emplace_back(loc_p[i], loc_p[j], loc_Mp_mat(i, j));
                }
                if (std::abs(loc_Kp_mat(i, j)) > 0.0) {
                    trip_Kp.emplace_back(loc_p[i], loc_p[j], loc_Kp_mat(i, j));
                }
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell_c; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell_c; ++j) {
                if (std::abs(loc_Mc_mat(i, j)) > 0.0) {
                    trip_Mc.emplace_back(loc_c[i], loc_c[j], loc_Mc_mat(i, j));
                }
                if (std::abs(loc_Kc_mat(i, j)) > 0.0) {
                    trip_Kc.emplace_back(loc_c[i], loc_c[j], loc_Kc_mat(i, j));
                }
                if (std::abs(loc_Mphi_mat(i, j)) > 0.0) {
                    trip_Mphi.emplace_back(loc_c[i], loc_c[j], loc_Mphi_mat(i, j));
                }
                if (std::abs(loc_KphiD_mat(i, j)) > 0.0) {
                    trip_KphiD.emplace_back(loc_c[i], loc_c[j], loc_KphiD_mat(i, j));
                }
            }
        }
    }

    M_d_.resize(n_d, n_d);
    M_d_.setFromTriplets(trip_Md.begin(), trip_Md.end());
    M_d_.makeCompressed();

    K0_mech_.resize(n_d, n_d);
    K0_mech_.setFromTriplets(trip_K0.begin(), trip_K0.end());
    K0_mech_.makeCompressed();

    Kvisco_mech_.resize(n_d, n_d);
    Kvisco_mech_.setFromTriplets(trip_Kvisco.begin(), trip_Kvisco.end());
    Kvisco_mech_.makeCompressed();

    B_biot_.resize(n_d, n_p);
    B_biot_.setFromTriplets(trip_B.begin(), trip_B.end());
    B_biot_.makeCompressed();

    M_p_.resize(n_p, n_p);
    M_p_.setFromTriplets(trip_Mp.begin(), trip_Mp.end());
    M_p_.makeCompressed();

    K_p_.resize(n_p, n_p);
    K_p_.setFromTriplets(trip_Kp.begin(), trip_Kp.end());
    K_p_.makeCompressed();

    M_c_.resize(n_c, n_c);
    M_c_.setFromTriplets(trip_Mc.begin(), trip_Mc.end());
    M_c_.makeCompressed();

    K_c_.resize(n_c, n_c);
    K_c_.setFromTriplets(trip_Kc.begin(), trip_Kc.end());
    K_c_.makeCompressed();

    M_phi_c_.resize(n_c, n_c);
    M_phi_c_.setFromTriplets(trip_Mphi.begin(), trip_Mphi.end());
    M_phi_c_.makeCompressed();
    K_phiD_c_.resize(n_c, n_c);
    K_phiD_c_.setFromTriplets(trip_KphiD.begin(), trip_KphiD.end());
    K_phiD_c_.makeCompressed();

    if (!verify_dof_and_mesh_consistency()) {
        throw std::runtime_error(
            "Fallita la verifica di coerenza tra DoF, support points e matrici invarianti.");
    }
}

bool InvariantFEMMatrices2D::verify_dof_and_mesh_consistency(double tol) const {
    if (dh_disp_.n_dofs() == 0 || dh_press_.n_dofs() == 0 || dh_tracer_.n_dofs() == 0) {
        return false;
    }

    if (fe_press_.degree == fe_tracer_.degree) {
        if (dh_press_.n_dofs() != dh_tracer_.n_dofs()) {
            return false;
        }
        const Eigen::MatrixXd pts_p = get_press_support_points();
        const Eigen::MatrixXd pts_c = get_tracer_support_points();
        if ((pts_p - pts_c).cwiseAbs().maxCoeff() > tol) {
            return false;
        }

        std::vector<dealii::types::global_dof_index> loc_p(fe_press_.n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> loc_c(fe_tracer_.n_dofs_per_cell());
        auto cp = dh_press_.begin_active();
        auto cc = dh_tracer_.begin_active();
        for (; cp != dh_press_.end(); ++cp, ++cc) {
            cp->get_dof_indices(loc_p);
            cc->get_dof_indices(loc_c);
            if (loc_p != loc_c) {
                return false;
            }
        }
    }

    const Eigen::VectorXd ones_p = Eigen::VectorXd::Ones(dh_press_.n_dofs());
    const Eigen::VectorXd ones_c = Eigen::VectorXd::Ones(dh_tracer_.n_dofs());
    if ((K_p_ * ones_p).cwiseAbs().maxCoeff() > tol) {
        return false;
    }
    if ((K_c_ * ones_c).cwiseAbs().maxCoeff() > tol) {
        return false;
    }

    double area_tria = 0.0;
    for (const auto& cell : tria_.active_cell_iterators()) {
        area_tria += cell->measure();
    }
    const double area_mc = ones_c.dot(M_c_ * ones_c);
    if (std::abs(area_tria - area_mc) / std::max(area_tria, 1.0e-12) > 1.0e-10) {
        return false;
    }

    const Eigen::VectorXd ones_d = Eigen::VectorXd::Ones(dh_disp_.n_dofs());
    const double area_md = 0.5 * ones_d.dot(M_d_ * ones_d);
    if (std::abs(area_tria - area_md) / std::max(area_tria, 1.0e-12) > 1.0e-10) {
        return false;
    }

    return true;
}

double InvariantFEMMatrices2D::compute_disp_L2_norm(const Eigen::VectorXd& d_dofs) const {
    if (static_cast<std::size_t>(d_dofs.size()) != dh_disp_.n_dofs()) {
        throw std::invalid_argument("Dimensione di d_dofs non coerente con num_disp_dofs.");
    }
    const double val2 = d_dofs.dot(M_d_ * d_dofs);
    return std::sqrt(std::max(0.0, val2));
}

double InvariantFEMMatrices2D::compute_press_L2_norm(const Eigen::VectorXd& p_dofs) const {
    if (static_cast<std::size_t>(p_dofs.size()) != dh_press_.n_dofs()) {
        throw std::invalid_argument("Dimensione di p_dofs non coerente con num_press_dofs.");
    }
    const double val2 = p_dofs.dot(M_p_ * p_dofs);
    return std::sqrt(std::max(0.0, val2));
}

double InvariantFEMMatrices2D::compute_tracer_L2_norm(const Eigen::VectorXd& c_dofs) const {
    if (static_cast<std::size_t>(c_dofs.size()) != dh_tracer_.n_dofs()) {
        throw std::invalid_argument("Dimensione di c_dofs non coerente con num_tracer_dofs.");
    }
    const double val2 = c_dofs.dot(M_c_ * c_dofs);
    return std::sqrt(std::max(0.0, val2));
}

Eigen::SparseMatrix<double> InvariantFEMMatrices2D::assemble_tracer_advection(
    const Eigen::VectorXd& p_dofs,
    bool supg_stabilization,
    double D_tracer) const {
    const std::size_t n_p = dh_press_.n_dofs();
    const std::size_t n_c = dh_tracer_.n_dofs();
    if (static_cast<std::size_t>(p_dofs.size()) != n_p) {
        throw std::invalid_argument("Dimensione di p_dofs non coerente con num_press_dofs.");
    }

    dealii::Vector<double> p_dealii(n_p);
    for (std::size_t i = 0; i < n_p; ++i) {
        p_dealii[i] = p_dofs(static_cast<Eigen::Index>(i));
    }

    const unsigned int quad_deg = std::max(fe_press_.degree, fe_tracer_.degree) + 2;
    dealii::QGauss<2> quadrature(quad_deg);
    dealii::FEValues<2> fe_val_p(fe_press_, quadrature, dealii::update_gradients);
    dealii::FEValues<2> fe_val_c(
        fe_tracer_, quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int dofs_per_cell_c = fe_tracer_.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature.size();
    std::vector<dealii::Tensor<1, 2>> grad_p_q(n_q_points);
    std::vector<dealii::types::global_dof_index> loc_c(dofs_per_cell_c);

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    triplets.reserve(tria_.n_active_cells() * dofs_per_cell_c * dofs_per_cell_c);

    auto cell_p = dh_press_.begin_active();
    auto cell_c = dh_tracer_.begin_active();
    for (; cell_c != dh_tracer_.end(); ++cell_p, ++cell_c) {
        fe_val_p.reinit(cell_p);
        fe_val_c.reinit(cell_c);
        fe_val_p.get_function_gradients(p_dealii, grad_p_q);
        cell_c->get_dof_indices(loc_c);

        const double h_K = cell_c->diameter();
        const auto& mat = material(static_cast<unsigned int>(cell_c->material_id()));
        Eigen::MatrixXd loc_A = Eigen::MatrixXd::Zero(dofs_per_cell_c, dofs_per_cell_c);

        for (unsigned int q = 0; q < n_q_points; ++q) {
            const dealii::Tensor<1, 2> q_darcy = -(physics_.K_h * mat.hydraulic_mobility) * grad_p_q[q];
            const double q_norm = q_darcy.norm();
            const double jxw = fe_val_c.JxW(q);

            double tau_supg = 0.0;
            if (supg_stabilization && q_norm > 1.0e-12 && D_tracer > 0.0) {
                const double pe_K = (q_norm * h_K) /
                    (2.0 * physics_.phi_0 * mat.porosity * mat.tracer_diffusivity * D_tracer);
                if (pe_K > 1.0) {
                    tau_supg = (h_K / (2.0 * q_norm)) * (1.0 - 1.0 / pe_K);
                }
            }

            for (unsigned int i = 0; i < dofs_per_cell_c; ++i) {
                const auto grad_z_i = fe_val_c.shape_grad(i, q);
                const double q_dot_grad_z = q_darcy * grad_z_i;
                for (unsigned int j = 0; j < dofs_per_cell_c; ++j) {
                    const double phi_c_j = fe_val_c.shape_value(j, q);
                    double val = -(phi_c_j * q_dot_grad_z) * jxw;
                    if (tau_supg > 0.0) {
                        const auto grad_c_j = fe_val_c.shape_grad(j, q);
                        val += tau_supg * (q_darcy * grad_c_j) * q_dot_grad_z * jxw;
                    }
                    loc_A(i, j) += val;
                }
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell_c; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell_c; ++j) {
                if (std::abs(loc_A(i, j)) > 0.0) {
                    triplets.emplace_back(loc_c[i], loc_c[j], loc_A(i, j));
                }
            }
        }
    }

    Eigen::SparseMatrix<double> A_adv(n_c, n_c);
    A_adv.setFromTriplets(triplets.begin(), triplets.end());
    A_adv.makeCompressed();
    return A_adv;
}

double InvariantFEMMatrices2D::compute_max_darcy_velocity(const Eigen::VectorXd& p_dofs) const {
    const std::size_t n_p = dh_press_.n_dofs();
    if (static_cast<std::size_t>(p_dofs.size()) != n_p) {
        throw std::invalid_argument("Dimensione di p_dofs non coerente con num_press_dofs.");
    }

    dealii::Vector<double> p_dealii(n_p);
    for (std::size_t i = 0; i < n_p; ++i) {
        p_dealii[i] = p_dofs(static_cast<Eigen::Index>(i));
    }

    dealii::QGauss<2> quadrature(fe_press_.degree + 1);
    dealii::FEValues<2> fe_val_p(fe_press_, quadrature, dealii::update_gradients);
    std::vector<dealii::Tensor<1, 2>> grad_p_q(quadrature.size());

    double max_q = 0.0;
    for (const auto& cell_p : dh_press_.active_cell_iterators()) {
        fe_val_p.reinit(cell_p);
        fe_val_p.get_function_gradients(p_dealii, grad_p_q);
        const auto& mat = material(static_cast<unsigned int>(cell_p->material_id()));
        for (const auto& gp : grad_p_q) {
            const double q_mag = physics_.K_h * mat.hydraulic_mobility * gp.norm();
            if (q_mag > max_q) {
                max_q = q_mag;
            }
        }
    }
    return max_q;
}

Eigen::MatrixXd InvariantFEMMatrices2D::get_cell_centers() const {
    Eigen::MatrixXd centers(tria_.n_active_cells(), 2);
    Eigen::Index idx = 0;
    for (const auto& cell : tria_.active_cell_iterators()) {
        const auto c = cell->center();
        centers(idx, 0) = c[0];
        centers(idx, 1) = c[1];
        ++idx;
    }
    return centers;
}

Eigen::MatrixXd InvariantFEMMatrices2D::compute_cell_darcy_fluxes(const Eigen::VectorXd& p_dofs) const {
    const std::size_t n_p = dh_press_.n_dofs();
    if (static_cast<std::size_t>(p_dofs.size()) != n_p) {
        throw std::invalid_argument("Dimensione di p_dofs non coerente con num_press_dofs.");
    }

    dealii::Vector<double> p_dealii(n_p);
    for (std::size_t i = 0; i < n_p; ++i) {
        p_dealii[i] = p_dofs(static_cast<Eigen::Index>(i));
    }

    dealii::QMidpoint<2> q_mid;
    dealii::FEValues<2> fe_val_p(fe_press_, q_mid, dealii::update_gradients);
    std::vector<dealii::Tensor<1, 2>> grad_p_mid(1);

    Eigen::MatrixXd fluxes(tria_.n_active_cells(), 2);
    Eigen::Index idx = 0;
    for (const auto& cell_p : dh_press_.active_cell_iterators()) {
        fe_val_p.reinit(cell_p);
        fe_val_p.get_function_gradients(p_dealii, grad_p_mid);
        const auto& mat = material(static_cast<unsigned int>(cell_p->material_id()));
        const dealii::Tensor<1, 2> q_vec =
            -(physics_.K_h * mat.hydraulic_mobility) * grad_p_mid[0];
        fluxes(idx, 0) = q_vec[0];
        fluxes(idx, 1) = q_vec[1];
        ++idx;
    }
    return fluxes;
}

static Eigen::MatrixXd extract_support_points_helper(const dealii::DoFHandler<2>& dh) {
    const std::map<dealii::types::global_dof_index, dealii::Point<2>> support_points =
        dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<2>(), dh);
    Eigen::MatrixXd out(static_cast<Eigen::Index>(dh.n_dofs()), 2);
    for (const auto& [dof_idx, pt] : support_points) {
        out(static_cast<Eigen::Index>(dof_idx), 0) = pt[0];
        out(static_cast<Eigen::Index>(dof_idx), 1) = pt[1];
    }
    return out;
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

Eigen::MatrixXd InvariantFEMMatrices2D::get_mesh_vertices() const {
    const auto& verts = tria_.get_vertices();
    Eigen::MatrixXd out(verts.size(), 2);
    for (std::size_t i = 0; i < verts.size(); ++i) {
        out(static_cast<Eigen::Index>(i), 0) = verts[i][0];
        out(static_cast<Eigen::Index>(i), 1) = verts[i][1];
    }
    return out;
}

Eigen::MatrixXi InvariantFEMMatrices2D::get_cell_quad_connectivity() const {
    Eigen::MatrixXi quads(tria_.n_active_cells(), 4);
    Eigen::Index c_idx = 0;
    for (const auto& cell : tria_.active_cell_iterators()) {
        quads(c_idx, 0) = static_cast<int>(cell->vertex_index(0));
        quads(c_idx, 1) = static_cast<int>(cell->vertex_index(1));
        quads(c_idx, 2) = static_cast<int>(cell->vertex_index(3));
        quads(c_idx, 3) = static_cast<int>(cell->vertex_index(2));
        ++c_idx;
    }
    return quads;
}

Eigen::MatrixXi InvariantFEMMatrices2D::get_cell_tracer_dof_connectivity() const {
    Eigen::MatrixXi dof_quads(tria_.n_active_cells(), 4);
    std::vector<dealii::types::global_dof_index> loc_c(fe_tracer_.n_dofs_per_cell());
    Eigen::Index c_idx = 0;
    for (const auto& cell_c : dh_tracer_.active_cell_iterators()) {
        cell_c->get_dof_indices(loc_c);
        dof_quads(c_idx, 0) = static_cast<int>(loc_c[0]);
        dof_quads(c_idx, 1) = static_cast<int>(loc_c[1]);
        dof_quads(c_idx, 2) = static_cast<int>(loc_c[3]);
        dof_quads(c_idx, 3) = static_cast<int>(loc_c[2]);
        ++c_idx;
    }
    return dof_quads;
}

Eigen::VectorXi InvariantFEMMatrices2D::get_cell_material_ids() const {
    Eigen::VectorXi mat_ids(tria_.n_active_cells());
    Eigen::Index c_idx = 0;
    for (const auto& cell : tria_.active_cell_iterators()) {
        mat_ids(c_idx) = static_cast<int>(cell->material_id());
        ++c_idx;
    }
    return mat_ids;
}

} // namespace reteco