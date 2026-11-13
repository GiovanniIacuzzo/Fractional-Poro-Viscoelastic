# Fractional Poro-Viscoelasticity

**Tracer-based identifiability of fractional mechanical memory · Finite elements · Inverse problems · Computational verification**

This repository contains a research implementation of a coupled fractional poro-viscoelastic and tracer-transport model. A C++20 finite-element solver built on [deal.II](https://www.dealii.org/) is connected to Python workflows for numerical verification, observation design, local identifiability, synthetic inverse experiments, and publication-quality visualization.

The central question is not merely whether a tracer signal *changes* with mechanical memory, but whether that change is **distinguishable from compensation by other parameters and from numerical-model discrepancy**.

> [!IMPORTANT]
> **Research software — synthetic-data studies.** The current implementation supports methodological investigations, not clinical diagnosis or patient-specific parameter estimation. The BraTS-based experiments use MRI-derived **2D geometry** with prescribed material contrasts and **synthetic tracer observations**. A full-rank Jacobian, successful optimization, or a visually accurate fit is not by itself evidence of global identifiability.

## Contents

- [Research questions](#research-questions)
- [Mathematical model](#mathematical-model)
- [Scientific diagnostics and selected findings](#scientific-diagnostics-and-selected-findings)
- [Software architecture](#software-architecture)
- [Requirements and installation](#requirements-and-installation)
- [Running experiments](#running-experiments)
- [Reproducibility and numerical integrity](#reproducibility-and-numerical-integrity)
- [Data, limitations, and responsible interpretation](#data-limitations-and-responsible-interpretation)
- [Citation and licensing](#citation-and-licensing)

## Research questions

The software is designed to investigate four connected questions:

1. **Transmission of information:** Under which forcing and flow regimes does mechanical memory affect concentration-only observations through the mechanics–pressure–Darcy-flow–tracer pathway?
2. **Parameter-specific identifiability:** How much information about the mechanical fractional order $\beta$ remains after the other unknown coefficients are allowed to compensate for it?
3. **Observation design:** Can a sensor placement that improves the *global* condition number nevertheless be less informative about $\beta$?
4. **Numerical discrepancy:** When are differences between forward discretizations comparable with, or larger than, the tracer response associated with a change in $\beta$?

These are **protocol-dependent inverse-problem questions**. The repository does not presume that a positive flow always guarantees identifiability, that a small residual implies accurate parameter recovery, or that local sensitivity constitutes global uniqueness.

## Mathematical model

The dimensionless model combines quasi-static Biot poromechanics, a fractional Kelvin–Voigt contribution to the solid stress, Darcy flow, and conservative fractional tracer transport:

$$
\begin{aligned}
-\nabla\!\cdot\boldsymbol\sigma &= \mathbf f_s,\\
\boldsymbol\sigma
  &= E(\mathbf x)\,\mathbb C_0:\boldsymbol\varepsilon(\mathbf d)
   + A_v\,\mathbb C_0:{}^{C}D_t^{\beta}\boldsymbol\varepsilon(\mathbf d)
   - bp\mathbf I,\\
\partial_t\!\left(S_p p+b\nabla\!\cdot\mathbf d\right)
  + \nabla\!\cdot\mathbf q &= s_f,\\
\mathbf q &= -\mathbb K_h(\mathbf x)\nabla p,\\
\phi(\mathbf x)\,{}^{C}D_t^{\alpha}c
  + \nabla\!\cdot\!\left(c\mathbf q-\phi(\mathbf x)D(\mathbf x)\nabla c\right)
  &=s_c.
\end{aligned}
$$

Here $\mathbf d$ denotes solid displacement, $p$ pore pressure, $\mathbf q$ Darcy flux, and $c$ tracer concentration. The Caputo orders $\beta$ and $\alpha$ describe **different memory mechanisms**: viscoelastic stress and tracer accumulation, respectively. The viscoelastic amplitude is $A_v=\eta_v\tau_v^{\beta}$ under the adopted constitutive convention. Regional material factors may modify $E$, $\mathbb K_h$, $D$, and $\phi$.

The implemented coupling is **one-way**: mechanics and hydraulics determine the velocity driving tracer transport; the tracer does not feed back onto the mechanical subsystem. The physical interpretation and dimensional scaling of the fractional tracer law depend on its explicitly stated characteristic-time convention; see `configs/scales_and_physics.yaml` and the associated manuscript.

The principal inverse parameter vector is

$$
\boldsymbol\theta=(\alpha,\beta,D,E)^{\mathsf T},
$$

with an extended six-parameter analysis that also treats $K_h$ and $A_v$ as unknowns. The observation operator samples tracer concentration at prescribed positions and times:

$$
\mathbf y=\mathcal G_{h,\mathcal S}(\boldsymbol\theta)
+\boldsymbol\varepsilon.
$$

The index $h$ denotes the spatial/temporal discretization and $\mathcal S$ the fixed observation design. The reference values, bounds, loads, time grids, and noise conventions belong to the **individual experiment protocol** and must not be silently transferred between experiments.

## Scientific diagnostics and selected findings

### Conditional information about mechanical memory

Let $\widetilde J=W\,\partial_{\mathbf z}\mathcal G$ be the weighted observation Jacobian with respect to **normalized parameter coordinates** $\mathbf z$. Separate the column $j_\beta$ from the remaining columns $J_{-\beta}$. The component of the $\beta$ sensitivity that cannot be reproduced, at first order, by changes in the other parameters is

$$
 r_\beta=(I-P_{-\beta})j_\beta,
 \qquad
 s_{\beta\mid-\beta}=\|r_\beta\|_2,
$$

where $P_{-\beta}$ is the orthogonal projector onto the range of $J_{-\beta}$. Under the adopted independent-Gaussian, common-scale observation weighting, the corresponding local conditional information is $s_{\beta\mid-\beta}^{\,2}$.

**Illustrative numerical result (regional anatomical model, 16 sensors and 5 times per design):**

| Unknowns | Sensor design | Global $\kappa_2(\widetilde J)$ | $s_{\beta\mid-\beta}$ |
|:--|:--|--:|--:|
| $\alpha,\beta,D,E$ | Balanced | 42.85 | 0.59659 |
| $\alpha,\beta,D,E$ | Edema-focused | 38.04 | 0.30285 |
| $\alpha,\beta,D,E,K_h,A_v$ | Balanced | 58.67 | 0.55835 |
| $\alpha,\beta,D,E,K_h,A_v$ | Edema-focused | 103.26 | 0.16739 |

Both designs use the **same absolute reference noise scale** for this comparison. In the four-parameter case the edema-focused design improves the global condition number, yet retains only about **25.8%** of the balanced design's local conditional information on $\beta$. This is a **specific numerical finding**, not a universal ranking of sensor placements or anatomies. The six-parameter extension exposes further compensation by hydraulic and viscoelastic coefficients.

Complementary diagnostics include singular values and numerical rank, finite-difference step audits, fixed-parameter cost slices, and nonlinear reoptimized profiles. A converged profile point is not automatically a global optimizer or a statistically calibrated confidence interval.

### Discretization sensitivity

The anatomical workflows compare distinct meshes and time grids. Experiment `20` evaluates a **shared spatial material field at quadrature points** across three nominal mesh resolutions (3, 2, and 1.5 mm), reducing the confounding effect of changing cell-wise material labels. Mesh boundaries are still approximated differently.

Three resolution levels support **discrepancy comparisons**, but not a defensible claim of asymptotic convergence order for this anatomical experiment. Comparing the *norms* of a cross-mesh discrepancy and a $\beta$-induced response indicates potential confounding; it does **not** establish that the vectors align or predict a parameter bias. A directional-bias prediction validated against nonlinear cross-mesh inversions is a distinct research step.

### Distinct noise protocols

The nominal percentage $\eta$ is not interchangeable across all experiments:

- **Synthetic square:** $\sigma=\eta_{\infty}\|\mathbf y^\dagger\|_{\infty}$.
- **Anatomical studies:** $\sigma=\eta_{\mathrm{RMS}}\,\operatorname{RMS}(\mathbf y^\dagger_{\mathrm{balanced}})$ for the common-scale balanced/edema comparison.

For a common interpretation, report $\sigma/\operatorname{RMS}(\mathbf y^\dagger)$ and, where relevant, the realized noise ratio $\|\boldsymbol\varepsilon\|_2/\|\mathbf y^\dagger\|_2$. Geometry, sampling operators, regularization, and weighting may also differ between protocols; comparisons across them are **not controlled one-factor experiments**.

## Software architecture

| Layer | Implementation | Responsibility |
|:--|:--|:--|
| Native forward model | C++20, deal.II, pybind11 | Coupled FEM assembly/solution, fractional history, diagnostics |
| Scientific workflows | Python 3.11, NumPy, SciPy | Parameter sweeps, Jacobians, inversion, benchmarks |
| Anatomy and meshing | nibabel, Gmsh, meshio, scikit-image | MRI-derived 2D supports, quadrilateral meshes, material maps |
| Results and visualization | JSON/CSV/NPZ, Matplotlib | Inspectable experiment records and vector PDF figures |
| Verification | pytest and dedicated numerical studies | Unit, integration, numerical, and protocol checks |

```text
.
├── CMakeLists.txt                 # Native build and Python bindings
├── Makefile                       # Convenience targets
├── environment.yml                # Pinned Python major/minor and package constraints
├── pyproject.toml                 # Python package metadata
├── prepare.sh                     # Environment setup, provenance, optional build
├── configs/                       # Physics, discretization, anatomy, benchmark protocols
├── cpp_core/
│   ├── include/reteco/            # Solver interfaces and numerical components
│   ├── src/                        # FEM, memory, monitors, manufactured solutions
│   └── bindings/                   # pybind11 interface
├── python/reteco/
│   ├── anatomy/                    # BraTS geometry and fixed material field
│   ├── forward/                    # Forward evaluation and synthetic targets
│   ├── geometry/                   # Mesh preparation
│   ├── identifiability/            # Jacobians, SVD, profile diagnostics
│   ├── inverse/                    # Objectives, CPSO, baselines
│   ├── observation/                # Sampling and noise models
│   └── visualization/              # Scientific plotting utilities
├── experiments/                   # Numbered, independently configured campaigns
├── scripts/                       # Focused setup, audit and anatomical runners
├── replot/                         # Figures regenerated from archived results
└── tests/                          # Numerical and workflow tests
```

**Generated at runtime, not included in the source snapshot:** `build/`, `artifacts/`, MRI data under `data/`, and `replot/output/` figures. The manuscript is maintained separately from this code-only snapshot; do not assume that a `paper/` directory or a completed dataset is present after cloning.

## Requirements and installation

The provided setup has been developed around **macOS / Apple Silicon**. Other platforms require a compatible native deal.II build and have not been asserted here to be fully tested.

- Python **3.11**, a C++20 compiler, CMake, Ninja, and a Conda/Mamba-compatible environment manager.
- **deal.II 9.5.1 or newer**, installed separately (the setup script supports a `candi` installation).
- Eigen, pybind11, NumPy/SciPy, and the dependencies declared in `environment.yml`.
- For anatomical workflows: Gmsh, meshio, nibabel, and the separately obtained BraTS input volumes.

Clone the repository (access may be restricted during manuscript development):

```bash
git clone https://github.com/GiovanniIacuzzo/Fractional-Poro-Viscoelastic.git
cd Fractional-Poro-Viscoelastic
```

After installing deal.II with a compiler and architecture compatible with the intended environment:

```bash
# The script creates .conda_env and activate_reteco.sh, then builds reteco._core.
DEAL_II_DIR="/absolute/path/to/deal.II" ./prepare.sh --build

source ./activate_reteco.sh
python -c "import reteco._core; print('Native extension available')"
pytest tests/ -q
```

If deal.II is under `~/dealii-candi/deal.II-*`, `prepare.sh` can discover it without `DEAL_II_DIR`. The script writes `artifacts/environment_provenance.json` and `artifacts/conda_lock.json` for local provenance tracking. After editing C++ sources, rebuild the extension:

```bash
cmake --build build --parallel
```

A native extension built for another system or an earlier source revision is not a substitute for rebuilding; in particular, the `.so` bundled in a local macOS snapshot is not portable.

## Running experiments

All commands below are executed from the repository root after `source ./activate_reteco.sh`. Some studies depend on previous artifacts or BraTS meshes and **will fail explicitly if their inputs are missing**. Use `python <script> --help` to inspect the full protocol before allocating significant compute time.

### 1 — Forward verification

```bash
python experiments/01_run_forward_verification.py
```

Manufactured solutions, fractional-memory checks, balance diagnostics, and refinement studies are provided by the numbered experiments and `tests/` suite. Passing a selected test is evidence for that test configuration, not a global convergence theorem.

### 2 — Synthetic-square identifiability

```bash
python experiments/02_run_identifiability_study.py \
  --output-dir artifacts/02_identifiability \
  --eps-step 1e-4 \
  --profiles
```

This produces Jacobian/SVD diagnostics, one-at-a-time sections, and—when `--profiles` is specified—reoptimized objective profiles. The profile option performs substantially more forward evaluations than the basic SVD analysis.

### 3 — MRI-derived anatomical forward problem

After obtaining the externally licensed BraTS volumes specified in `configs/brats_anatomical_forward.yaml`:

```bash
python scripts/run_brats_anatomical_pipeline.py \
  --config configs/brats_anatomical_forward.yaml

python scripts/run_anatomical_forward_simulation.py \
  --config configs/brats_anatomical_forward.yaml
```

The geometry report and mesh quality checks should be inspected before anatomical inverse studies. MRI segmentation determines geometry and prescribed material contrasts; it does **not** supply observed tracer time series or calibrated patient-specific material coefficients.

### 4 — Anatomical inverse and robustness campaigns

The numbered workflows support distinct scientific tasks:

| Experiments | Focus |
|:--|:--|
| `03`–`12` | Synthetic inverse recovery, objective diagnostics, CPSO and baseline sensitivity, $\beta$ profiles, mesh sensitivity |
| `13`–`15` | Flow-excitation identifiability, inverse controls, fractional-model variants |
| `16` | Anatomical Jacobians, four-/six-parameter identifiability, sensor designs |
| `17` | Nonlinear $\beta$ profiles with other parameters reoptimized |
| `18` | Cross-discretization synthetic targets and FEM inverse controls (cell-assigned material protocol) |
| `19` | Noise-dependent nonlinear FEM Monte Carlo with multistart and restart support |
| `20` | Three-level anatomical forward comparison using one shared quadrature-point material field |

For example, an **existing** experiment `19` run can be resumed only with the *exact* configuration recorded in its `manifest.json`:

```bash
python experiments/19_run_anatomical_noise_monte_carlo.py \
  --out-dir artifacts/19_montecarlo_extended \
  --resume \
  --max-new-replicates 0 \
  --design balanced \
  --noise-fractions 0.001 0.005 0.01 0.05 \
  --replicates 8 --starts 2 --max-nfev 30 \
  --fd-step 0.0005 --jac forward --seed 20261111
```

This is a **continuation example, not a fresh-run command**. It requires the original input files, matching native code/configuration hashes, and a preexisting manifest. For a new campaign use a new `--out-dir` without `--resume`, choose the sampling budget in advance, and preserve all attempted replicas (including unsuccessful ones). Do not merge independent campaigns solely because they share a random seed.

Experiment `20` exposes separate material-field generation and FEM evaluation stages:

```bash
python experiments/20_run_anatomical_fixed_material_crossmesh.py --mode field
python experiments/20_run_anatomical_fixed_material_crossmesh.py --mode run
```

It requires the frozen anatomical reference, segmentation, and three meshes. It is a forward discrepancy study—not yet a completed nonlinear cross-mesh **inverse** validation with the same continuous material field.

### Regenerating figures

```bash
python replot/run_all_replots.py
```

`replot/` reconstructs vector PDFs and high-resolution PNGs from saved experiment data without rerunning FEM solves. The master command expects the relevant artifacts; missing inputs cause an error rather than fabricated plots. Before manuscript use, verify figure provenance and check that plots generated from older artifacts have been refreshed after any solver/configuration change.

## Reproducibility and numerical integrity

A result should be traceable to **its code revision, numerical protocol, inputs, and solver diagnostics**, not only to a figure.

For every reported experiment, retain:

1. **Source identity:** Git commit SHA, compiler/deal.II/Python versions, and any changes to the native extension.
2. **Protocol:** full YAML/CLI settings, target and inverse discretizations, material representation, sensor coordinates, observation times, normalization, weighting, regularization, parameter bounds, and random seeds.
3. **Data identity:** hashes or frozen copies of input meshes, material fields, target observations, and any external anatomical source used.
4. **Execution evidence:** complete logs; solver statuses, budgets, gradient/optimality diagnostics where available; attempted, failed and convergent replicate counts.
5. **Machine-readable outputs:** the JSON, CSV, and NPZ reports used to derive each table or figure.

Where provided by a workflow, `manifest.json` and `--resume` protect against mixing incompatible campaigns. Do not manually overwrite old result directories or silently relabel values when the source protocol has changed.

**Interpretation rules:**

- *Numerical rank* is relative to a declared threshold, parameter normalization and observation operator.
- *Local identifiability* is not equivalent to *global identifiability*.
- *Optimizer termination* does not certify global optimality or reliable parameter recovery.
- *Three-level mesh fits* do not establish an asymptotic order when geometry and error sources are not controlled.
- *A discrepancy magnitude* alone does not show that model error lies in the conditional $\beta$-sensitivity direction.
- *Monte Carlo estimates* should state their sample size, convergence selection, and uncertainty; small pilots do not establish precise failure rates.

## Data, limitations, and responsible interpretation

**BraTS data.** MRI modalities and segmentations are obtained separately from the original provider, subject to the dataset's terms. They are **not redistributed here**. Avoid committing medical images or derived files that might violate their access or privacy conditions.

**Scope.** The anatomical cases are two-dimensional computational proxies. Material contrasts, loads, tracer concentrations, and observation noise are prescribed for controlled experiments; they are not clinical measurements. Physical coefficients in exploratory numerical regimes should not be presented as validated brain-tissue estimates.

**Open research steps.** Directional prediction and nonlinear validation of mesh-induced $\beta$ bias using a common physical material field, broader observation-design validation, and larger noise ensembles are separate from the completed local sensitivity analyses. They should be reported as completed results only after the underlying simulations and numerical audits are archived.

## Citation and licensing

**Citation.** The associated manuscript is under revision. Please cite a **versioned public manuscript and/or archived software release once available**, along with the exact repository commit used. No publication DOI or archival release identifier is claimed here.

**Software license.** The supplied source snapshot did not include a `LICENSE` file. An explicit software license should be added before public release; absent such a file, this README does not grant rights to reuse or redistribute the code. Licenses for third-party libraries and BraTS data are independent of any future software license.

---

**Research implementation:** Giovanni Giuseppe Iacuzzo  
**Field:** Fractional poromechanics, inverse problems, numerical analysis, and scientific computing  
**Repository:** [Fractional-Poro-Viscoelastic](https://github.com/GiovanniIacuzzo/Fractional-Poro-Viscoelastic)
