#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "reteco/parameters.hpp"
#include "reteco/time_grid.hpp"
#include "reteco/caputo_l1_memory.hpp"
#include "reteco/invariant_matrices.hpp"
#include "reteco/initial_compatibility.hpp"
#include "reteco/biot_visco_solver.hpp"
#include "reteco/tracer_transport_solver.hpp"
#include "reteco/energy_mass_monitor.hpp"
#include "reteco/mms_manufactured.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "RETECO v1 C++20 deal.II High-Performance Forward Core";

    py::class_<reteco::ParameterTheta>(m, "ParameterTheta")
        .def(py::init<>())
        .def(py::init([](double alpha, double beta, double D, double E) {
            reteco::ParameterTheta t{alpha, beta, D, E};
            t.validate();
            return t;
        }), py::arg("alpha"), py::arg("beta"), py::arg("D"), py::arg("E"))
        .def_readwrite("alpha", &reteco::ParameterTheta::alpha)
        .def_readwrite("beta", &reteco::ParameterTheta::beta)
        .def_readwrite("D", &reteco::ParameterTheta::D)
        .def_readwrite("E", &reteco::ParameterTheta::E)
        .def("validate", &reteco::ParameterTheta::validate);

    py::class_<reteco::FixedPhysicsConfig>(m, "FixedPhysicsConfig")
        .def(py::init<>())
        .def_readwrite("nu", &reteco::FixedPhysicsConfig::nu)
        .def_readwrite("b", &reteco::FixedPhysicsConfig::b)
        .def_readwrite("S_p", &reteco::FixedPhysicsConfig::S_p)
        .def_readwrite("K_h", &reteco::FixedPhysicsConfig::K_h)
        .def_readwrite("eta_v", &reteco::FixedPhysicsConfig::eta_v)
        .def_readwrite("tau_v", &reteco::FixedPhysicsConfig::tau_v)
        .def_readwrite("phi_0", &reteco::FixedPhysicsConfig::phi_0)
        .def_readwrite("plane_strain", &reteco::FixedPhysicsConfig::plane_strain)
        .def_readwrite("L_0", &reteco::FixedPhysicsConfig::L_0)
        .def_readwrite("T_0", &reteco::FixedPhysicsConfig::T_0)
        .def_readwrite("tau_c", &reteco::FixedPhysicsConfig::tau_c)
        .def_readwrite("E_0", &reteco::FixedPhysicsConfig::E_0)
        .def_readwrite("c_ref", &reteco::FixedPhysicsConfig::c_ref)
        .def("validate", &reteco::FixedPhysicsConfig::validate)
        .def("mu_0", &reteco::FixedPhysicsConfig::mu_0)
        .def("lambda_0", &reteco::FixedPhysicsConfig::lambda_0);

    py::class_<reteco::TimeGrid>(m, "TimeGrid")
        .def(py::init<const std::vector<double>&>())
        .def_static("make_graded", &reteco::TimeGrid::make_graded,
                    py::arg("T"), py::arg("num_steps"), py::arg("r") = 1.0)
        .def("num_steps", &reteco::TimeGrid::num_steps)
        .def("time_at", &reteco::TimeGrid::time_at)
        .def("dt", &reteco::TimeGrid::dt)
        .def("times", &reteco::TimeGrid::times)
        .def("is_uniform", &reteco::TimeGrid::is_uniform, py::arg("rtol") = 1.0e-12);

    py::class_<reteco::CaputoL1Memory>(m, "CaputoL1Memory")
        .def(py::init<double, const reteco::TimeGrid&, std::size_t>(),
             py::arg("gamma_order"), py::arg("grid"), py::arg("num_dofs"))
        .def("reset", &reteco::CaputoL1Memory::reset)
        .def("current_implicit_coeff", &reteco::CaputoL1Memory::current_implicit_coeff)
        .def("step_weights", &reteco::CaputoL1Memory::step_weights)
        .def("compute_history_load", &reteco::CaputoL1Memory::compute_history_load)
        .def("evaluate_discrete_derivative", &reteco::CaputoL1Memory::evaluate_discrete_derivative)
        .def("commit_step", &reteco::CaputoL1Memory::commit_step);

    py::class_<reteco::InvariantFEMMatrices2D>(m, "InvariantFEMMatrices2D")
        .def(py::init<unsigned int, const reteco::FixedPhysicsConfig&,
                      unsigned int, unsigned int, unsigned int, bool>(),
             py::arg("subdivisions_N"),
             py::arg("physics"),
             py::arg("fe_deg_disp") = 2,
             py::arg("fe_deg_press") = 1,
             py::arg("fe_deg_tracer") = 1,
             py::arg("dirichlet_all_displacement_boundary") = true)
        .def("num_disp_dofs", &reteco::InvariantFEMMatrices2D::num_disp_dofs)
        .def("num_press_dofs", &reteco::InvariantFEMMatrices2D::num_press_dofs)
        .def("num_tracer_dofs", &reteco::InvariantFEMMatrices2D::num_tracer_dofs)
        .def("K0_mech", &reteco::InvariantFEMMatrices2D::K0_mech)
        .def("B_biot", &reteco::InvariantFEMMatrices2D::B_biot)
        .def("M_p", &reteco::InvariantFEMMatrices2D::M_p)
        .def("K_p", &reteco::InvariantFEMMatrices2D::K_p)
        .def("M_c", &reteco::InvariantFEMMatrices2D::M_c)
        .def("K_c", &reteco::InvariantFEMMatrices2D::K_c)
        .def("assemble_tracer_advection", &reteco::InvariantFEMMatrices2D::assemble_tracer_advection,
             py::arg("p_dofs"), py::arg("supg_stabilization") = false, py::arg("D_tracer") = 0.0)
        .def("compute_max_darcy_velocity", &reteco::InvariantFEMMatrices2D::compute_max_darcy_velocity)
        .def("get_disp_support_points", &reteco::InvariantFEMMatrices2D::get_disp_support_points)
        .def("get_press_support_points", &reteco::InvariantFEMMatrices2D::get_press_support_points)
        .def("get_tracer_support_points", &reteco::InvariantFEMMatrices2D::get_tracer_support_points);

    py::class_<reteco::InitialCompatibility>(m, "InitialCompatibility")
        .def_static("compute_compatible_d0", &reteco::InitialCompatibility::compute_compatible_d0)
        .def_static("compatibility_residual_norm", &reteco::InitialCompatibility::compatibility_residual_norm);

    py::class_<reteco::BiotViscoStepSolver>(m, "BiotViscoStepSolver")
        .def(py::init<const reteco::InvariantFEMMatrices2D&,
                      const reteco::ParameterTheta&,
                      const reteco::TimeGrid&>())
        .def("reset", &reteco::BiotViscoStepSolver::reset)
        .def("advance_step", [](reteco::BiotViscoStepSolver& self,
                                std::size_t n,
                                const Eigen::VectorXd& F_mech_n,
                                const Eigen::VectorXd& S_fluid_n) {
            Eigen::VectorXd d_n, p_n;
            self.advance_step(n, F_mech_n, S_fluid_n, d_n, p_n);
            return py::make_tuple(d_n, p_n);
        });

    py::class_<reteco::TracerTransportStepSolver>(m, "TracerTransportStepSolver")
        .def(py::init<const reteco::InvariantFEMMatrices2D&,
                      const reteco::ParameterTheta&,
                      const reteco::TimeGrid&,
                      bool>(),
             py::arg("fem"), py::arg("theta"), py::arg("grid"), py::arg("use_supg") = false)
        .def("reset", &reteco::TracerTransportStepSolver::reset)
        .def("advance_step", [](reteco::TracerTransportStepSolver& self,
                                std::size_t n,
                                const Eigen::VectorXd& p_n,
                                const Eigen::VectorXd& S_tracer_n) {
            Eigen::VectorXd c_n;
            self.advance_step(n, p_n, S_tracer_n, c_n);
            return c_n;
        });

    py::class_<reteco::StepDiagnostics>(m, "StepDiagnostics")
        .def_readonly("time", &reteco::StepDiagnostics::time)
        .def_readonly("elastic_energy", &reteco::StepDiagnostics::elastic_energy)
        .def_readonly("fluid_storage_energy", &reteco::StepDiagnostics::fluid_storage_energy)
        .def_readonly("total_energy", &reteco::StepDiagnostics::total_energy)
        .def_readonly("cumulative_visco_dissip", &reteco::StepDiagnostics::cumulative_visco_dissip)
        .def_readonly("cumulative_darcy_dissip", &reteco::StepDiagnostics::cumulative_darcy_dissip)
        .def_readonly("cumulative_external_work", &reteco::StepDiagnostics::cumulative_external_work)
        .def_readonly("energy_balance_residual", &reteco::StepDiagnostics::energy_balance_residual)
        .def_readonly("step_numerical_dissipation", &reteco::StepDiagnostics::step_numerical_dissipation)
        .def_readonly("step_discrete_identity_error", &reteco::StepDiagnostics::step_discrete_identity_error)
        .def_readonly("tracer_total_mass", &reteco::StepDiagnostics::tracer_total_mass);

    py::class_<reteco::EnergyMassMonitor>(m, "EnergyMassMonitor")
        .def(py::init<const reteco::InvariantFEMMatrices2D&>())
        .def("compute_tracer_mass", &reteco::EnergyMassMonitor::compute_tracer_mass)
        .def("compute_mechanical_fluid_energy", &reteco::EnergyMassMonitor::compute_mechanical_fluid_energy)
        .def("evaluate_trajectory_diagnostics", &reteco::EnergyMassMonitor::evaluate_trajectory_diagnostics);

    py::class_<reteco::MMSErrorReport>(m, "MMSErrorReport")
        .def_readonly("h_mesh", &reteco::MMSErrorReport::h_mesh)
        .def_readonly("dt_max", &reteco::MMSErrorReport::dt_max)
        .def_readonly("d_L2", &reteco::MMSErrorReport::d_L2)
        .def_readonly("d_H1", &reteco::MMSErrorReport::d_H1)
        .def_readonly("p_L2", &reteco::MMSErrorReport::p_L2)
        .def_readonly("p_H1", &reteco::MMSErrorReport::p_H1)
        .def_readonly("c_L2", &reteco::MMSErrorReport::c_L2)
        .def_readonly("c_H1", &reteco::MMSErrorReport::c_H1);

    py::class_<reteco::MMSManufacturedSystem>(m, "MMSManufacturedSystem")
        .def(py::init<const reteco::FixedPhysicsConfig&,
                      const reteco::ParameterTheta&,
                      double>(),
             py::arg("physics"), py::arg("theta"), py::arg("delta_tracer") = 0.4)
        .def("run_full_mms_test", &reteco::MMSManufacturedSystem::run_full_mms_test,
             py::arg("subdivisions_N"),
             py::arg("grid"),
             py::arg("fe_deg_disp") = 2,
             py::arg("fe_deg_press") = 1,
             py::arg("fe_deg_tracer") = 1);
}