# RETECO

## Fractional Poro-Viscoelasticity and Tracer-Based Parameter Identifiability

**RETECO** is a research codebase for the numerical study of tracer transport in a fractional poro-viscoelastic Biot-type medium. It combines a finite-element forward solver implemented in **C++20 / deal.II** with Python workflows for model configuration, numerical verification, local identifiability analysis, synthetic inverse experiments, and result visualization.

The repository accompanies the research manuscript *Tracer-Based Parameter Identifiability in Fractional Poro-Viscoelasticity*. Its focus is methodological: establish what the implemented model and observation protocol can identify, verify the numerical components used in the experiments, and document the computational evidence without overstating it as clinical validation.

> **Research status.** This is research software under active development, not a validated clinical application. The inverse studies described here use synthetic observations. The BraTS-based workflow builds an anatomical 2D computational domain for forward simulations; it does not estimate patient-specific parameters from clinical measurements.

**Repository access:** the repository is currently private during scientific development and manuscript review. Access therefore requires authorization. A public, versioned release should be tied to a frozen code revision, configuration files, and the corresponding experiment records before the work is submitted or archived.

---

## Scientific scope

### Model

The implemented model combines three components:

1. **Quasi-static poro-viscoelastic mechanics.** The solid displacement and pore pressure follow a Biot-type formulation with a fractional Kelvin–Voigt contribution of order \(\beta\).
2. **Darcy flow.** The fluid flux is computed from the pressure field, \(\mathbf q=-\mathbb K_h\nabla p\).
3. **Passive fractional tracer transport.** The concentration follows a Caputo-type sub-diffusive transport equation of order \(\alpha\), driven by the computed Darcy flux.

The coupling is **one-way**: the mechanical–hydraulic subsystem determines the flow that transports the tracer; tracer concentration does not feed back into the mechanical or hydraulic equations.

A schematic form of the dimensionless system is

\[
\begin{aligned}
-\nabla\!\cdot\boldsymbol\sigma &= \mathbf f_s,\\
\boldsymbol\sigma &= E\,\mathbb C_0(\nu):\boldsymbol\varepsilon(\mathbf d)
+\eta_v\tau_v^\beta\mathbb C_0(\nu):{}^C D_t^\beta\boldsymbol\varepsilon(\mathbf d)
-bp\mathbf I,\\
\partial_t(S_p p+b\nabla\!\cdot\mathbf d)+\nabla\!\cdot\mathbf q &= s_f,\\
\mathbf q&=-\mathbb K_h\nabla p,\\
\phi_0{}^C D_t^\alpha c+\nabla\!\cdot(c\mathbf q)-\phi_0D\Delta c &= s_c.
\end{aligned}
\]

The four parameters studied in the inverse problem are

\[
\boldsymbol\theta=(\alpha,\beta,D,E)^{\mathsf T},
\]

where \(\alpha\) is the tracer fractional order, \(\beta\) is the viscoelastic fractional order, \(D\) is the dimensionless tracer diffusivity, and \(E\) is the dimensionless drained elastic modulus. Other physical coefficients and characteristic scales are defined in the YAML configuration files.

### Numerical and inferential methods

- **Fractional time discretization:** Caputo L1 convolution weights on uniform and graded/non-uniform time grids.
- **Finite elements:** the current solver and anatomical pipeline are two-dimensional; the mechanics use a \(Q_2/Q_1\)-type displacement–pressure pair, while the tracer is represented with \(Q_1\) elements.
- **Verification:** manufactured-solution studies, checks of initial mechanical compatibility, discrete energy/mass diagnostics, and mesh/time-step sensitivity studies.
- **Identifiability diagnostics:** normalized observation Jacobians, singular-value decomposition (SVD), numerical-rank assessment, local geometric correlation diagnostics, and one-at-a-time objective slices.
- **Inverse experiments:** regularized least-squares objectives, the implemented continuous particle-swarm optimizer (CPSO), multi-seed and noise studies, and comparisons with standard optimization baselines.
- **Anatomical workflow:** controlled extraction of a 2D quadrilateral mesh from BraTS 2020 MRI volumes, followed by forward simulation with explicitly configured synthetic forcing.

The optimized one-dimensional objective profiles in this repository are **numerical conditional objective profiles**. They are not statistical profile-likelihood curves and do not, by themselves, provide confidence intervals.

---

## What the experiments can—and cannot—establish

The workflow is designed to separate numerical evidence from stronger scientific claims:

- In the zero-flow setting, when only tracer concentration is observed, changes in \(E\) and \(\beta\) may leave the tracer observation map unchanged. The corresponding zero Jacobian columns provide a structural non-identifiability diagnostic for that protocol.
- With active flow, the code evaluates whether the local observation Jacobian has full numerical rank at a specified parameter vector and for a specified observation design. This is **local, point-dependent evidence**, not proof of global identifiability.
- Manufactured-solution tests and algebraic balance monitors verify selected numerical components under specified test conditions. They do not constitute a general proof of stability or convergence for every geometry, forcing, parameter regime, or mesh.
- Anatomical mesh metrics quantify agreement with the processed support mask. They do not establish histological correctness or clinical validity.
- Synthetic twin experiments evaluate parameter recovery under declared targets, noise levels, seeds, objective weights, and evaluation budgets. Their outcomes should not be interpreted as performance on patient-derived observations.

These distinctions are central to interpreting the manuscript and reproducing its results.

---

## Repository map

```bash
.
├── CMakeLists.txt                 # C++ core and Python extension build
├── environment.yml                # Conda/Mamba scientific Python environment
├── prepare.sh                     # Environment setup and optional native build
├── Makefile                       # Common build, test, and experiment targets
├── configs/                       # Versioned physics and experiment configurations
├── cpp_core/
│   ├── include/reteco/            # C++ interfaces and model components
│   ├── src/                        # Fractional memory, FEM, solvers, diagnostics
│   └── bindings/                   # pybind11 interface
├── python/reteco/
│   ├── anatomy/                    # BraTS input checks and anatomy processing
│   ├── forward/                    # Forward solver orchestration and target generation
│   ├── geometry/                   # Geometry and mesh helpers
│   ├── identifiability/             # Jacobian, SVD, and objective diagnostics
│   ├── inverse/                    # CPSO, objective, regularization, baselines
│   ├── observation/                 # Observation/sensor operators
│   └── visualization/               # Publication figures and diagnostic plots
├── experiments/                    # Reproducible experiment entry points
├── scripts/                        # Anatomical pipeline and focused run scripts
├── tests/                           # Numerical, structural, and workflow tests
└── paper/
    ├── paper.tex                   # Manuscript entry point
    ├── sections/                   # Main manuscript sections
    ├── appendices/                 # Supplementary methods and diagnostics
    ├── images/                     # Vector figures used by the manuscript
    ├── references.bib              # Bibliography
    └── paper.pdf                   # Tracked PDF snapshot
```

Runtime-generated files (including local build products, data, and experiment artifacts) are intentionally excluded from version control. Preserve the relevant output files when archiving a run; they are not automatically part of the Git history.

## Requirements

The supplied setup script is designed around a macOS development workflow. The native core uses **C++20** and **deal.II**; the Python environment is declared in `environment.yml`.

| Component | Requirement / role |
|---|---|
| Python | 3.11 |
| C++ compiler | C++20-capable compiler; Apple Clang is selected by `prepare.sh` on macOS |
| deal.II | Installed separately; `prepare.sh` checks for version 9.5.1 or later |
| CMake | 3.27 or later in the declared Conda environment; the project CMake file requires 3.24 or later |
| Ninja | Native build generator |
| Eigen | 3.4 or later |
| pybind11 | C++/Python bindings |
| Gmsh, meshio, nibabel, scikit-image | Anatomical mesh and MRI processing |
| NumPy, SciPy, SymPy, Pydantic, PyYAML, Matplotlib, pytest | Numerical workflows, configuration validation, plots, and tests |

`deal.II` is not installed by Conda in this setup. Install it separately (the setup script documents the [candi](https://github.com/dealii/candi) workflow), and make sure its compiler and architecture are compatible with the Python environment and the rest of the native dependencies. On Apple Silicon, avoid mixing `arm64` and `x86_64` builds.

## Installation and build

Clone the repository using an account that has access:

```bash
git clone https://github.com/GiovanniIacuzzo/Fractional-Poro-Viscoelastic.git
cd Fractional-Poro-Viscoelastic
```

If `deal.II` is installed in the default location searched by `prepare.sh`, run:

```bash
./prepare.sh --build
```

Otherwise, provide its installation prefix explicitly:

```bash
DEAL_II_DIR="/absolute/path/to/deal.II" ./prepare.sh --build
```

The script creates or updates the local Conda environment, writes `activate_reteco.sh`, checks the `deal.II` installation, configures CMake, compiles the `reteco._core` extension, and installs the Python package in editable mode. The generated activation script also sets conservative numerical thread defaults used by the project workflow.

Activate the environment in each new terminal session:

```bash
source ./activate_reteco.sh
```

To rebuild after changing C++ sources:

```bash
cmake --build build --parallel
```

If the native dependencies cannot be found, check `DEAL_II_DIR`, the output of `cmake`, and the architecture/compiler consistency before changing Python dependencies.

## Tests and numerical verification

Run the complete test suite after building and activating the environment:

```bash
source ./activate_reteco.sh
pytest tests/ -q
```

A focused test can be run independently, for example:

```bash
pytest tests/test_01_caputo_l1.py -q
```

The suite includes checks for fractional L1 weights and limiting behaviour, initial compatibility, energy/mass diagnostics, manufactured-solution convergence, the zero-flow identifiability case, anatomical processing and mesh integration, inverse-problem consistency, objective-function modes, and experiment-protocol audits.

**Do not infer that every test passes from the presence of a test file.** This repository does not currently publish a cross-platform continuous-integration guarantee. Record the complete test log, compiler/deal.II versions, and commit hash for the environment on which the final experiments are run.

## Running the main workflows

The Makefile provides convenient entry points from the repository root. Run them from an activated environment:

```bash
source ./activate_reteco.sh
make verify-forward
make identifiability
make twin-robustness
make sota-benchmark
```

| Target | Purpose |
|---|---|
| `make verify-forward` | Run the forward numerical-verification workflow |
| `make identifiability` | Run the structural/local identifiability study |
| `make twin-robustness` | Run the configured synthetic twin-experiment workflow |
| `make sota-benchmark` | Run the configured comparison between optimization methods; this name refers to the repository target, not a claim of exhaustive state-of-the-art coverage |
| `make test` | Build the native extension and run `pytest tests/` |

### Configuration files

| File | Purpose |
|---|---|
| `configs/scales_and_physics.yaml` | Characteristic scales and fixed physical parameters shared by the workflows |
| `configs/mms_benchmark.yaml` | Manufactured-solution verification and refinement studies |
| `configs/twin_experiment_synthetic.yaml` | Synthetic target/inverse discretizations, observations, forcing, noise, and objective settings |
| `configs/brats_anatomical_forward.yaml` | BraTS input provenance checks, 2D mesh construction, anatomical forward-simulation settings, and mesh sensitivity |
| `configs/sota_comparison.yaml` | Optimizer-comparison settings and evaluation budgets |

Treat the configuration used for an experiment as part of the result. Before comparing runs, verify that the target, inverse discretization, objective mode, weighting, regularization, random seeds, and forward-solver evaluation budget match the intended protocol.

## BraTS 2020 anatomical workflow

The anatomical workflow is intentionally separate from the synthetic inverse experiments. It expects the BraTS 2020 subject configured in `configs/brats_anatomical_forward.yaml` and the five NIfTI volumes listed there (`T1`, `T1ce`, `T2`, `FLAIR`, and segmentation). By default, the inputs are expected beneath `data/01_raw/`.

First, run the anatomical preprocessing and quadrilateral-mesh construction:

```bash
source ./activate_reteco.sh
python scripts/run_brats_anatomical_pipeline.py \
  --config configs/brats_anatomical_forward.yaml
```

After the geometry report confirms that the mandatory checks passed, run the forward simulation:

```bash
python scripts/run_anatomical_forward_simulation.py \
  --config configs/brats_anatomical_forward.yaml
```

The configuration checks the expected image grid and spacing and forbids implicit realignment for the specified input volumes. Missing or invalid inputs cause an explicit failure; the production script does not silently replace missing MRI data with synthetic volumes.

The pipeline derives a **2D computational domain** from a selected MRI slice and produces a quadrilateral mesh. In the associated forward experiment, the loads and tracer initial conditions are configured synthetic inputs, with their locations guided by the processed anatomy. This workflow is intended for geometry-aware numerical experiments—not clinical prediction, diagnosis, or inverse estimation from patient measurements.

### Data and artifacts

MRI volumes are not distributed in this repository. Obtain any required dataset from its official source and comply with its access terms, license, attribution, and redistribution restrictions. Do not commit raw or identifiable medical data.

Generated outputs are written to the configured `artifacts/` directories, which are ignored by Git. For each reported experiment, retain at least:

- the Git commit SHA and exact YAML configuration;
- the environment/provenance record generated by setup, where available;
- random seeds and objective/optimizer settings;
- full console logs and machine-readable reports (JSON/CSV);
- the generated figures and any data hashes used to identify a frozen target.

The setup script creates environment metadata locally; because generated artifacts are not version-controlled by default, they must be deliberately archived with a release or supplementary materials if they are needed to reproduce a result.

## Manuscript

The LaTeX manuscript and its supporting figures are under `paper/`:

- `paper/paper.tex` — document entry point;
- `paper/sections/` — main text;
- `paper/appendices/` — numerical, anatomical, identifiability, and reproducibility details;
- `paper/images/` — vector figures;
- `paper/references.bib` — bibliography;
- `paper/paper.pdf` — a tracked PDF snapshot, which may not always reflect the newest source changes during active development.

With a LaTeX installation providing the packages used by the manuscript and `latexmk`, compile from the paper directory:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode paper.tex
```

When preparing a submission, compile the PDF from the exact source revision intended for release and verify the generated figures, citations, bibliography, and appendices visually. Archive the source revision and final PDF together.

## Reproducibility and interpretation

A reproducible numerical claim should be traceable to a specific configuration and run, not just to a plot. The project therefore uses explicit YAML configurations, parameter bounds and coordinate transformations, random seeds, numerical diagnostics, and distinct target/inverse discretizations in the synthetic twin experiments. The different discretizations are intended to reduce the classical *inverse crime*; they do not, by themselves, remove all discretization error or guarantee inverse-problem convergence.

For any comparison or reported number, document which protocol generated it. Do not interpret numerical rank as proof of global identifiability, empirical spread across a small number of random seeds as a confidence interval, objective-profile thresholds as statistical intervals, or synthetic anatomical simulations as evidence of clinical efficacy.

## License

The software is distributed under the MIT License; see [`LICENSE`](LICENSE). The license of the code does not override the terms governing external datasets, including BraTS MRI data.

## Citation

The repository is currently private and does not yet identify a public, immutable software release or archival DOI. When the manuscript and code are frozen for public release, add the corresponding citation metadata and version/commit identifier here. Until then, cite the associated manuscript according to the citation details supplied by its authors.

---

*Maintainer: Giovanni Giuseppe Iacuzzo*  
*Project: RETECO — fractional poro-viscoelastic forward modelling and tracer-based parameter-identifiability research.*
