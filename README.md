# Fractional Poro Viscoelastic

## Directory:

```bash
├── 📁 configs
│   ├── ⚙️ mms_benchmark.yaml
│   ├── ⚙️ scales_and_physics.yaml
│   ├── ⚙️ sota_comparison.yaml
│   └── ⚙️ twin_experiment_synthetic.yaml
├── 📁 cpp_core
│   ├── 📁 bindings
│   │   └── ⚡ pybind_module.cpp
│   ├── 📁 include
│   │   └── 📁 reteco
│   │       ├── ⚡ biot_visco_solver.hpp
│   │       ├── ⚡ caputo_l1_memory.hpp
│   │       ├── ⚡ energy_mass_monitor.hpp
│   │       ├── ⚡ initial_compatibility.hpp
│   │       ├── ⚡ invariant_matrices.hpp
│   │       ├── ⚡ mms_manufactured.hpp
│   │       ├── ⚡ parameters.hpp
│   │       ├── ⚡ time_grid.hpp
│   │       └── ⚡ tracer_transport_solver.hpp
│   └── 📁 src
│       ├── ⚡ biot_visco_solver.cpp
│       ├── ⚡ caputo_l1_memory.cpp
│       ├── ⚡ energy_mass_monitor.cpp
│       ├── ⚡ initial_compatibility.cpp
│       ├── ⚡ invariant_matrices.cpp
│       ├── ⚡ mms_manufactured.cpp
│       └── ⚡ tracer_transport_solver.cpp
├── 📁 experiments
│   ├── 🐍 01_run_forward_verification.py
│   ├── 🐍 02_run_identifiability_study.py
│   ├── 🐍 03_run_twin_robustness.py
│   └── 🐍 04_run_sota_benchmark.py
├── 📁 paper
│   ├── 📁 images
│   ├── 📁 sections
│   │   ├── 📄 section_1_introduction.tex
│   │   ├── 📄 section_2_forward.tex
│   │   ├── 📄 section_3_numerical.tex
│   │   ├── 📄 section_4_inverse.tex
│   │   ├── 📄 section_5_results.tex
│   │   └── 📄 section_6_conclusions.tex
│   ├── 📕 paper.pdf
│   ├── 📄 paper.tex
│   └── 📄 references.bib
├── 📁 python
│   └── 📁 reteco
│       ├── 📁 forward
│       │   ├── 🐍 target_generator.py
│       │   └── 🐍 wrapper.py
│       ├── 📁 geometry
│       │   ├── 🐍 gmsh_generator.py
│       │   └── 🐍 mri_segmentation.py
│       ├── 📁 identifiability
│       │   ├── 🐍 profile_likelihood.py
│       │   ├── 🐍 sensitivity_jacobian.py
│       │   └── 🐍 svd_conditioning.py
│       ├── 📁 inverse
│       │   ├── 🐍 cpso.py
│       │   ├── 🐍 objective_functional.py
│       │   ├── 🐍 regularization.py
│       │   └── 🐍 sota_runners.py
│       ├── 📁 observation
│       │   ├── 🐍 noise_model.py
│       │   └── 🐍 operator.py
│       ├── 📁 visualization
│       │   ├── 🐍 benchmark_plots.py
│       │   ├── 🐍 identifiability_plots.py
│       │   └── 🐍 verification_plots.py
│       ├── 🐍 __init__.py
│       └── 🐍 config_schema.py
├── 📁 tests
│   ├── 🐍 test_01_caputo_l1.py
│   ├── 🐍 test_02_initial_compatibility.py
│   ├── 🐍 test_03_energy_and_mass.py
│   ├── 🐍 test_04_mms_coupled.py
│   ├── 🐍 test_05_identifiability_zero_q.py
│   └── 🐍 test_06_inverse_consistency.py
├── ⚙️ .gitignore
├── 📄 CMakeLists.txt
├── 📄 Makefile
├── 📝 README.md
├── ⚙️ environment.yml
├── 📄 prepare.sh
└── ⚙️ pyproject.toml
```
