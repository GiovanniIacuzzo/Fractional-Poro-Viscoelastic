#!/usr/bin/env python3
"""Opt-in, fail-closed patch for quadrature-point anatomical materials.

Run from repository root after reviewing the changes. No GitHub operations.
The patch has no effect unless RETECO_QP_MATERIAL_FIELD is explicitly set
BEFORE the first InvariantFEMMatrices2D object is instantiated in a process.

Use `--check` to inspect applicability and `--apply` to write local source files.
A .orig-pre-qp file is created beside each changed source; patch never silently
replaces an already-modified file.
"""
import argparse
import difflib
from pathlib import Path

HPP = Path('cpp_core/include/reteco/invariant_matrices.hpp')
CPP = Path('cpp_core/src/invariant_matrices.cpp')

CPP_STRUCT = r'''
namespace {
// One globally specified physical coefficient field, independent of cell IDs.
// The field file is produced from the ORIGINAL canonical BraTS segmentation
// by experiments/20_run_anatomical_fixed_material_crossmesh.py.
struct QPMaterialGrid {
    unsigned int nx = 0, ny = 0;
    double x0 = 0.0, y0 = 0.0, dx = 0.0, dy = 0.0;
    std::vector<reteco::RegionMaterialMultipliers> values;

    explicit QPMaterialGrid(const std::string& name) {
        std::ifstream input(name);
        if (!input) throw std::runtime_error("Cannot open RETECO_QP_MATERIAL_FIELD: " + name);
        std::string magic;
        input >> magic >> nx >> ny >> x0 >> y0 >> dx >> dy;
        if (!input || magic != "RETECO_QP_MATERIAL_V1" || nx < 2 || ny < 2 ||
            !(dx > 0.0) || !(dy > 0.0) || nx > 20000u || ny > 20000u) {
            throw std::runtime_error("Invalid quadrature material field header: " + name);
        }
        values.resize(static_cast<std::size_t>(nx) * ny);
        for (unsigned int j = 0; j < ny; ++j) {
            for (unsigned int i = 0; i < nx; ++i) {
                auto& v = values[static_cast<std::size_t>(j) * nx + i];
                input >> v.stiffness >> v.hydraulic_mobility >> v.tracer_diffusivity >> v.porosity;
                if (!input) throw std::runtime_error("Truncated quadrature material field: " + name);
                v.validate(0.20); // strict positivity; physical porosity checked in FEM configuration
            }
        }
        double trailing = 0.0;
        if (input >> trailing) throw std::runtime_error("Extra values in quadrature material field: " + name);
        if (!input.eof()) throw std::runtime_error("Malformed quadrature material field: " + name);
    }

    reteco::RegionMaterialMultipliers at(const dealii::Point<2>& p) const {
        const double xi = (p[0] - x0) / dx;
        const double yi = (p[1] - y0) / dy;
        // Out-of-grid evaluation is an error, rather than an unnoticed extrapolation.
        constexpr double slack = 1.e-7;
        if (xi < -slack || yi < -slack || xi > nx-1+slack || yi > ny-1+slack) {
            throw std::runtime_error("FEM quadrature point is outside common BraTS material raster");
        }
        const double xx = std::max(0.0, std::min(xi, static_cast<double>(nx-1)));
        const double yy = std::max(0.0, std::min(yi, static_cast<double>(ny-1)));
        const unsigned int i0 = std::min(static_cast<unsigned int>(xx), nx-2);
        const unsigned int j0 = std::min(static_cast<unsigned int>(yy), ny-2);
        const double a = xx-i0, b = yy-j0;
        const auto& A = values[static_cast<std::size_t>(j0)*nx+i0];
        const auto& B = values[static_cast<std::size_t>(j0)*nx+i0+1];
        const auto& C = values[static_cast<std::size_t>(j0+1)*nx+i0];
        const auto& D = values[static_cast<std::size_t>(j0+1)*nx+i0+1];
        const auto blend = [&](double va, double vb, double vc, double vd) {
            return (1-a)*(1-b)*va + a*(1-b)*vb + (1-a)*b*vc + a*b*vd;
        };
        reteco::RegionMaterialMultipliers result;
        result.stiffness = blend(A.stiffness,B.stiffness,C.stiffness,D.stiffness);
        result.hydraulic_mobility = blend(A.hydraulic_mobility,B.hydraulic_mobility,C.hydraulic_mobility,D.hydraulic_mobility);
        result.tracer_diffusivity = blend(A.tracer_diffusivity,B.tracer_diffusivity,C.tracer_diffusivity,D.tracer_diffusivity);
        result.porosity = blend(A.porosity,B.porosity,C.porosity,D.porosity);
        return result;
    }
};

const QPMaterialGrid* active_qp_material_grid() {
    // Loaded only once: caller MUST set the environment BEFORE constructing
    // the first FEM object. Use a fresh process for another physical field.
    static const std::unique_ptr<QPMaterialGrid> grid = []() -> std::unique_ptr<QPMaterialGrid> {
        const char* file = std::getenv("RETECO_QP_MATERIAL_FIELD");
        return (file && *file) ? std::make_unique<QPMaterialGrid>(file) : nullptr;
    }();
    return grid.get();
}
} // anonymous namespace
'''


def sub1(s, old, new, label):
    k = s.count(old)
    if k != 1:
        raise ValueError(f'[{label}] expected one matching block, found {k}; source changed?')
    return s.replace(old, new, 1)


def transform(h, c):
    h = sub1(h, '    const RegionMaterialMultipliers& material(unsigned int id) const;',
             '    const RegionMaterialMultipliers& material(unsigned int id) const;\n'
             '    RegionMaterialMultipliers material_at(const dealii::Point<2>& p, unsigned int id) const;', 'header method')
    c = sub1(c, '#include <fstream>\n', '#include <fstream>\n#include <cstdlib>\n#include <memory>\n#include <string>\n', 'includes')
    c = sub1(c, 'namespace reteco {', CPP_STRUCT + '\nnamespace reteco {', 'helper')
    ix = c.index('InvariantFEMMatrices2D::InvariantFEMMatrices2D(')
    # Insert before the first constructor, avoiding edits in constructor logic.
    c = c[:ix] + '''RegionMaterialMultipliers InvariantFEMMatrices2D::material_at(
    const dealii::Point<2>& p, unsigned int id) const {
    if (const auto* common_field = active_qp_material_grid()) {
        auto coeff = common_field->at(p);
        coeff.validate(physics_.phi_0);
        return coeff;
    }
    return material(id);
}

''' + c[ix:]
    c = sub1(c,
        'fe_disp_, quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);',
        'fe_disp_, quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);',
        'd quadrature updates')
    c_old = 'fe_tracer_, quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);'
    c_new = 'fe_tracer_, quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);'
    if c.count(c_old) != 2: raise ValueError('Expected exactly two tracer FEValues setups')
    c = c.replace(c_old, c_new)
    c = sub1(c,
        '        const auto& mat = material(static_cast<unsigned int>(cell_d->material_id()));\n', '',
        'cell constant material')
    c = sub1(c,
        '        for (unsigned int q = 0; q < n_q_points; ++q) {\n            const double jxw = fe_val_d.JxW(q);',
        '        for (unsigned int q = 0; q < n_q_points; ++q) {\n'
        '            const auto mat = material_at(fe_val_d.quadrature_point(q),\n'
        '                                         static_cast<unsigned int>(cell_d->material_id()));\n'
        '            const double jxw = fe_val_d.JxW(q);',
        'material at assembly qp')
    # Other operators, including Darcy advection and diagnostics.
    c = sub1(c,
        '        const auto& mat = material(static_cast<unsigned int>(cell_c->material_id()));\n'
        '        Eigen::MatrixXd loc_A',
        '        Eigen::MatrixXd loc_A', 'advection material const')
    c = sub1(c,
        '        for (unsigned int q = 0; q < n_q_points; ++q) {\n'
        '            const dealii::Tensor<1, 2> q_darcy',
        '        for (unsigned int q = 0; q < n_q_points; ++q) {\n'
        '            const auto mat = material_at(fe_val_c.quadrature_point(q),\n'
        '                                         static_cast<unsigned int>(cell_c->material_id()));\n'
        '            const dealii::Tensor<1, 2> q_darcy', 'advection material at qp')
    p_old = 'dealii::FEValues<2> fe_val_p(fe_press_, quadrature, dealii::update_gradients);'
    p_new = 'dealii::FEValues<2> fe_val_p(fe_press_, quadrature, dealii::update_gradients | dealii::update_quadrature_points);'
    if c.count(p_old) != 2: raise ValueError('Expected exactly two pressure FEValues setups')
    c = c.replace(p_old, p_new)
    c = sub1(c,
        '        const auto& mat = material(static_cast<unsigned int>(cell_p->material_id()));\n'
        '        for (const auto& gp : grad_p_q) {\n'
        '            const double q_mag',
        '        for (unsigned int q = 0; q < quadrature.size(); ++q) {\n'
        '            const auto mat = material_at(fe_val_p.quadrature_point(q),\n'
        '                                         static_cast<unsigned int>(cell_p->material_id()));\n'
        '            const auto& gp = grad_p_q[q];\n'
        '            const double q_mag', 'Darcy max material')
    c = sub1(c,
        'dealii::FEValues<2> fe_val_p(fe_press_, q_mid, dealii::update_gradients);',
        'dealii::FEValues<2> fe_val_p(fe_press_, q_mid, dealii::update_gradients | dealii::update_quadrature_points);',
        'midpoint updates')
    c = sub1(c,
        '        const auto& mat = material(static_cast<unsigned int>(cell_p->material_id()));\n'
        '        const dealii::Tensor<1, 2> q_vec',
        '        const auto mat = material_at(fe_val_p.quadrature_point(0),\n'
        '                                     static_cast<unsigned int>(cell_p->material_id()));\n'
        '        const dealii::Tensor<1, 2> q_vec', 'midpoint material')
    return h,c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true',help='Check matching source and show diff statistics, without writing')
    p.add_argument('--apply',action='store_true',help='Write locally, saving .orig-pre-qp backups')
    args = p.parse_args()
    if not (args.check or args.apply): p.error('Choose --check or --apply')
    root = Path(__file__).resolve().parents[1]
    ph, pc = root/HPP, root/CPP
    if not (ph.is_file() and pc.is_file()):
        raise FileNotFoundError('Place script in repository scripts/ and run there')
    h,c=ph.read_text(),pc.read_text()
    if 'RETECO_QP_MATERIAL_FIELD' in c:
        raise SystemExit('Already patched. No action; do not apply twice.')
    newh,newc=transform(h,c)
    print(f'Valid patch: {HPP} +{len(newh)-len(h)} chars; {CPP} +{len(newc)-len(c)} chars')
    if args.apply:
        for path,original,revised in ((ph,h,newh),(pc,c,newc)):
            backup=path.with_name(path.name+'.orig-pre-qp')
            if backup.exists(): raise FileExistsError(backup)
        for path,original,revised in ((ph,h,newh),(pc,c,newc)):
            backup=path.with_name(path.name+'.orig-pre-qp')
            backup.write_text(original)
            path.write_text(revised)
            print(f'[PATCHED] {path}; backup={backup}')
        print('Rebuild the native module (e.g. make build) before running experiment 20.')

if __name__=='__main__': main()
