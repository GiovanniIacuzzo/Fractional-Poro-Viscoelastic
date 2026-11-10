"""Guard the optional same-dose comparison against silent protocol changes."""
from pathlib import Path
import csv
import importlib.util
import json

import numpy as np
from scipy.sparse import diags

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reteco_phaseD_variants', ROOT/'experiments'/'15_run_fractional_model_variants.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Problem:
    def __init__(self):
        self.physics_cpp = type('Physics', (), {'phi_0': 0.2})()
        self.M_phi_c_csr = diags([1.0, 2.0], format='csr')
        self.num_tracer_dofs = 2


def test_initial_excess_dose_scaling():
    problem = Problem()
    c0 = np.array([1.1, 1.1])
    result = module._tracer_initial_dose(problem, c0, 0.1)
    assert np.isclose(result['storage_volume'], 0.6)
    assert np.isclose(result['excess_mass_above_baseline'], 0.6)
    factor = 0.75
    corrected = 0.1 + factor * (c0-0.1)
    out = module._tracer_initial_dose(problem, corrected, 0.1)
    assert np.isclose(out['excess_mass_above_baseline'], 0.45)
    assert np.isclose(out['total_stored_mass'], 0.45 + 0.1*0.6)


def _write_reference(tmp_path, mesh='mesh-hash', driver='driver-hash', physics='physics-hash'):
    manifest = {'domain':'anatomical','material_profile':'homogeneous',
                'mesh_sha256':mesh,'driving_sha256':driver,'physics_sha256':physics,
                'driving_metadata':{'initial_tracer_gaussian':{'baseline':0.1}}}
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest))
    with (tmp_path/'regional_time_series.csv').open('w', newline='') as f:
        out=csv.DictWriter(f,fieldnames=['model','time','storage_volume','stored_tracer_mass'])
        out.writeheader()
        out.writerow({'model':'M4','time':0,'storage_volume':1.5,'stored_tracer_mass':0.6})
        out.writerow({'model':'M4','time':0,'storage_volume':2.5,'stored_tracer_mass':0.9})
        out.writerow({'model':'M4','time':1,'storage_volume':4,'stored_tracer_mass':3})
    return tmp_path


def test_reference_excess_excludes_baseline(tmp_path):
    _write_reference(tmp_path)
    got=module._reference_excess_dose(tmp_path,baseline=0.1,mesh_sha='mesh-hash',
                                      driving_sha='driver-hash',physics_sha='physics-hash')
    assert np.isclose(got,1.1)


def test_reference_rejects_mismatching_mesh(tmp_path):
    _write_reference(tmp_path)
    import pytest
    with pytest.raises(ValueError,match='mesh_sha256'):
        module._reference_excess_dose(tmp_path,baseline=0.1,mesh_sha='another-mesh',
                                      driving_sha='driver-hash',physics_sha='physics-hash')
