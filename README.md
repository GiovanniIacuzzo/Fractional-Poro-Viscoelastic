# Modello Poro-Viscoelastico Frazionario

[![Standard C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![deal.II](https://img.shields.io/badge/deal.II-%3E%3D9.5.0-00599C.svg)](https://www.dealii.org/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![Build System](https://img.shields.io/badge/CMake-%3E%3D3.24%20%2B%20Ninja-064F8C.svg)](https://cmake.org/)
[![Riproducibilità](https://img.shields.io/badge/Riproducibilit%C3%A0-Seed%20%26%20Provenance%20Tracciati-2ea44f.svg)](#9-tracciabilit%C3%A0-criteri-anti-inverse-crime-e-limiti-metodologici)

Questo repository contiene l'implementazione di riferimento (C++20 / `deal.II` + Python) associata allo studio del problema diretto e del problema inverso per un mezzo poro-viscoelastico frazionario di tipo Biot accoppiato unidirezionalmente al trasporto sub-diffusivo di un tracciante passivo.

L'infrastruttura software è progettata con quattro obiettivi metodologici espliciti:
1. **Coerenza tra formulazione continua, adimensionalizzazione e discretizzazione**: separazione rigorosa tra il sottosistema meccanico-idraulico quasi-statico con memoria frazionaria di Kelvin–Voigt e l'equazione di advezione-diffusione frazionaria di Caputo per il tracciante.
2. **Verifica numerica indipendente del problema diretto**: test automatici dei pesi temporali $L1$ su griglie uniformi e graduate, controllo della condizione di compatibilità meccanica iniziale, monitoraggio dell'identità energetica discreta e della conservazione della massa del tracciante, e verifica tramite il Metodo delle Soluzioni Manufatte (MMS).
3. **Analisi di identificabilità strutturale e locale preliminare all'inversione**: valutazione numerica del Jacobiano normalizzato, dello spettro dei valori singolari (SVD) e verifica esplicita dei limiti di identificabilità di $(E, \beta)$ da sole osservazioni di concentrazione.
4. **Protocollo sperimentale inverso riproducibile ed esente da *inverse crime***: separazione tra la discretizzazione di generazione dei dati sintetici ($\mathcal{G}_{h_{\mathrm{tar}}}$) e la discretizzazione impiegata nell'inversione ($\mathcal{G}_{h_{\mathrm{inv}}}$), tracciamento immutabile del budget di valutazioni agli elementi finiti (FEM) e registrazione completa dei semi pseudocasuali e della provenienza dell'ambiente di calcolo.

---

## Indice

1. [Formulazione Matematica di Riferimento](#1-formulazione-matematica-di-riferimento)
2. [Discretizzazione Numerica Spazio-Temporale](#2-discretizzazione-numerica-spazio-temporale)
3. [Problema Inverso e Analisi di Identificabilità](#3-problema-inverso-e-analisi-di-identificabilit%C3%A0)
4. [Architettura del Repository](#4-architettura-del-repository)
5. [Acquisizione dei Dati MRI (Dataset BraTS) e Gestione `.gitignore`](#5-acquisizione-dei-dati-mri-dataset-brats-e-gestione-gitignore)
6. [Requisiti di Sistema e Installazione Deterministica](#6-requisiti-di-sistema-e-installazione-deterministica)
7. [Suite di Verifica Automatica (`pytest`)](#7-suite-di-verifica-automatica-pytest)
8. [Pipeline di Esecuzione degli Esperimenti e Output Generati](#8-pipeline-di-esecuzione-degli-esperimenti-e-output-generati)
9. [Tracciabilità, Criteri Anti-Inverse-Crime e Limiti Metodologici](#9-tracciabilit%C3%A0-criteri-anti-inverse-crime-e-limiti-metodologici)
10. [Citazione e Riferimenti Bibliografici](#10-citazione-e-riferimenti-bibliografici)

---

## 1. Formulazione Matematica di Riferimento

### 1.1. Variabili di stato e operatori frazionari

Sia $\Omega \subset \mathbb{R}^m$, con $m \in \{2, 3\}$, un dominio limitato con bordo Lipschitz $\partial\Omega = \overline{\Gamma_D} \cup \overline{\Gamma_N}$ e sia $t \in (0, T]$. Il modello descrive l'evoluzione di:
* $\mathbf{d}(\mathbf{x}, t)$: spostamento della matrice solida;
* $p(\mathbf{x}, t)$: pressione interstiziale dei pori;
* $\mathbf{q}(\mathbf{x}, t)$: flusso volumetrico di Darcy;
* $c(\mathbf{x}, t)$: concentrazione di un tracciante passivo.

Il tensore delle piccole deformazioni linearizzate è definito da:

$$
\boldsymbol{\varepsilon}(\mathbf{d}) = \frac{1}{2}\left(\nabla\mathbf{d} + \nabla\mathbf{d}^{\mathsf{T}}\right).
$$

Per un ordine frazionario $0 < \gamma < 1$, la derivata frazionaria nel senso di Caputo di una funzione assolutamente continua $g(t)$ è definita da:

$$
{}^C D_t^\gamma g(t) = \frac{1}{\Gamma(1-\gamma)} \int_0^t (t-s)^{-\gamma} g'(s)\,\mathrm{d}s = I^{1-\gamma} g'(t),
$$

dove $I^{1-\gamma}$ indica l'integrale frazionario di Riemann–Liouville. Per campi vettoriali o tensoriali l'operatore agisce componente per componente.

### 1.2. Adimensionalizzazione e sistema differenziale accoppiato

Fissate le scale caratteristiche di lunghezza $L_0$, tempo $T_0 = \tau_c$ (dove $\tau_c$ è la scala temporale associata al termine di accumulo frazionario del tracciante $\tau_c^{\alpha-1}{}^C D_t^\alpha c$), tensione $E_0$ e concentrazione $c_{\mathrm{ref}}$, i parametri adimensionali sono definiti tramite:

$$
E = \frac{E_{\mathrm{phys}}}{E_0}, \quad D = \frac{D_{\mathrm{phys}} T_0}{L_0^2}, \quad \eta_v = \frac{\eta_{v,\mathrm{phys}}}{E_0}, \quad \tau_v = \frac{\tau_{v,\mathrm{phys}}}{T_0}, \quad S_p = E_0 S_{p,\mathrm{phys}}, \quad \mathbb{K}_h = \frac{E_0 T_0}{L_0^2}\mathbb{K}_{h,\mathrm{phys}}.
$$

Con questa convenzione il coefficiente $D_{\mathrm{phys}}$ conserva l'unità fisica classica $L^2/T$, mentre la dipendenza dall'ordine $\alpha$ risiede nell'operatore di accumulo ${}^C D_t^\alpha c$ e nel fattore di scala $\tau_c^{\alpha-1}$ prima dell'adimensionalizzazione.

Il sistema adimensionale implementato nel nucleo computazionale è:

$$
\begin{aligned}
-\nabla \cdot \boldsymbol{\sigma} &= \mathbf{f}_s, \\
\boldsymbol{\sigma} &= E\mathbb{C}_0(\nu):\boldsymbol{\varepsilon}(\mathbf{d}) + \eta_v \tau_v^\beta \mathbb{C}_0(\nu):{}^C D_t^\beta \boldsymbol{\varepsilon}(\mathbf{d}) - b p \mathbf{I}, \\
\partial_t\left(S_p p + b\nabla \cdot \mathbf{d}\right) + \nabla \cdot \mathbf{q} &= s_f, \\
\mathbf{q} &= -\mathbb{K}_h \nabla p, \\
\phi_0 \, {}^C D_t^\alpha c + \nabla \cdot (c\mathbf{q}) - \phi_0 D \Delta c &= s_c,
\end{aligned}
$$

dove $\mathbb{C}_0(\nu)$ è il tensore costitutivo isotropo normalizzato (in ipotesi di deformazione piana nel caso bidimensionale, con $-1 < \nu < 1/2$), $b \in (0, 1]$ il coefficiente di Biot, $S_p > 0$ il coefficiente di immagazzinamento specifico e $\phi_0 \in (0, 1]$ la porosità di riferimento. L'accoppiamento è **unidirezionale**: il sottosistema poro-viscoelastico $(\mathbf{d}, p)$ determina il flusso di Darcy $\mathbf{q}$, il quale trasporta il tracciante $c$ senza retroazione della concentrazione sulle equazioni meccaniche o idrauliche.

Il vettore dei quattro parametri oggetto di identificazione è:

$$
\boldsymbol{\theta} = (\alpha, \beta, D, E)^{\mathsf{T}} \in \Theta = [\alpha_-, \alpha_+] \times [\beta_-, \beta_+] \times [D_-, D_+] \times [E_-, E_+],
$$

con $0 < \alpha_- < \alpha_+ < 1$, $0 < \beta_- < \beta_+ < 1$, $D_- > 0$ ed $E_- > 0$.

### 1.3. Condizioni al contorno, compatibilità iniziale e proprietà strutturali

* **Condizioni al contorno**:
  $$\mathbf{d} = \mathbf{0} \text{ su } \Gamma_D, \qquad \boldsymbol{\sigma}\mathbf{n} = \bar{\mathbf{t}} \text{ su } \Gamma_N, \qquad \mathbf{q}\cdot\mathbf{n} = 0 \text{ su } \partial\Omega, \qquad \mathbf{F}_c\cdot\mathbf{n} := (c\mathbf{q} - \phi_0 D\nabla c)\cdot\mathbf{n} = 0 \text{ su } \partial\Omega.$$
* **Condizione di compatibilità meccanica iniziale**: per soluzioni regolari tali che ${}^C D_t^\beta \boldsymbol{\varepsilon}(\mathbf{d}(t)) \to \mathbf{0}$ per $t \downarrow 0$, lo stato iniziale $(\mathbf{d}_0, p_0)$ deve soddisfare l'equilibrio elastico-porale a $t = 0$:
  $$a_E(\mathbf{d}_0, \mathbf{v}) - b(p_0, \nabla\cdot\mathbf{v})_\Omega = \langle F(0), \mathbf{v}\rangle \quad \forall \mathbf{v} \in V.$$
* **Identità energetica nella classe regolare**: posto $\mathcal{E}(t) = \frac{1}{2}a_E(\mathbf{d}(t), \mathbf{d}(t)) + \frac{S_p}{2}\|p(t)\|_{L^2(\Omega)}^2$ e $\mathcal{D}_\beta(t) = \int_0^t a_v({}^C D_s^\beta \mathbf{d}(s), \partial_s \mathbf{d}(s))\,\mathrm{d}s \ge 0$, vale l'identità a priori:
  $$\mathcal{E}(t) + \mathcal{D}_\beta(t) = \mathcal{E}(0) - \int_0^t a_h(p(s), p(s))\,\mathrm{d}s + \int_0^t \langle F(s), \partial_s\mathbf{d}(s)\rangle\,\mathrm{d}s + \int_0^t \langle s_f(s), p(s)\rangle\,\mathrm{d}s.$$
* **Conservazione della massa integrata del tracciante**: per $s_c = 0$ e flusso totale nullo al bordo ($\mathbf{F}_c \cdot \mathbf{n} = 0$), la massa spaziale integrata $M(t) = \int_\Omega c(\mathbf{x}, t)\,\mathrm{d}\mathbf{x}$ soddisfa ${}^C D_t^\alpha M(t) = 0$, da cui $M(t) = M(0)$ per ogni $t \in [0, T]$.

---

## 2. Discretizzazione Numerica Spazio-Temporale

### 2.1. Schema temporale $L1$ su griglie non uniformi

Sia $0 = t_0 < t_1 < \cdots < t_N = T$ una partizione temporale generale (ad esempio una griglia graduata $t_n = T(n/N)^r$ con esponente di graduazione $r \ge 1$) e passi $\Delta t_k = t_k - t_{k-1}$. L'operatore di Caputo di ordine $\gamma \in \{\alpha, \beta\}$ è approssimato mediante la formula $L1$ non uniforme:

$$
{}^C D_t^\gamma u(t_n) \approx \frac{1}{\Gamma(2-\gamma)} \sum_{k=1}^{n} \frac{u^k - u^{k-1}}{\Delta t_k} \left[(t_n - t_{k-1})^{1-\gamma} - (t_n - t_k)^{1-\gamma}\right].
$$

La classe C++ `reteco::CaputoL1Memory` calcola i pesi esatti per ciascun passo $t_n$ senza mai riutilizzare pesi uniformi quando $r > 1$, separando il coefficiente implicito corrente $d_{n,n}^{(\gamma)}$ dal termine di memoria storica dipendente da $\{u^0, \dots, u^{n-1}\}$.

### 2.2. Elementi finiti misti in `deal.II`

* **Sottosistema poro-viscoelastico $(\mathbf{d}_h, p_h)$**: discretizzato mediante coppia stabile di tipo Taylor–Hood su quadrilateri, con elementi biquadratici $[Q_2]^2$ (`FE_Q<2>(2)`) per lo spostamento $\mathbf{d}_h$ ed elementi bilineari $Q_1$ (`FE_Q<2>(1)`) per la pressione $p_h$.
* **Trasporto del tracciante $c_h$**: formulazione di Galerkin continua su spazio $Q_1$ (`FE_Q<2>(1)`). Qualora venga attivata l'opzione di stabilizzazione `use_supg = true`, essa viene dichiarata esplicitamente nei metadati dell'esperimento in quanto modifica l'operatore discreto.
* **Pre-assemblaggio delle matrici invarianti**: le matrici spaziali indipendenti dai parametri ($\mathbf{K}_{0,\mathrm{mech}}$, $\mathbf{B}_{\mathrm{biot}}$, $\mathbf{M}_p$, $\mathbf{K}_p$, $\mathbf{M}_c$, $\mathbf{K}_c$) sono assemblate una sola volta all'inizializzazione della mesh nella classe `reteco::InvariantFEMMatrices2D`, garantendo che ogni valutazione del problema diretto durante l'ottimizzazione richieda unicamente combinazioni lineari sparse e risoluzioni dei sistemi lineari.

---

## 3. Problema Inverso e Analisi di Identificabilità

### 3.1. Operatore osservativo e funzionale obiettivo

Detta $X_{\boldsymbol{\theta}, h} = (\mathbf{d}_{\boldsymbol{\theta}, h}, p_{\boldsymbol{\theta}, h}, c_{\boldsymbol{\theta}, h})$ la soluzione discreta del problema diretto e $\mathcal{H}$ l'operatore osservativo che seleziona i campi, i punti spaziali e gli istanti di campionamento, la mappa osservativa discreta è $\mathcal{G}_h(\boldsymbol{\theta}) = \mathcal{H}(X_{\boldsymbol{\theta}, h})$.

Nelle coordinate normalizzate $\widetilde{\boldsymbol{\theta}} = \psi(\boldsymbol{\theta}) \in [0, 1]^4$:

$$
\widetilde{\alpha} = \frac{\alpha - \alpha_-}{\alpha_+ - \alpha_-}, \quad
\widetilde{\beta} = \frac{\beta - \beta_-}{\beta_+ - \beta_-}, \quad
\widetilde{D} = \frac{\log D - \log D_-}{\log D_+ - \log D_-}, \quad
\widetilde{E} = \frac{\log E - \log E_-}{\log E_+ - \log E_-},
$$

la stima inversa minimizza il funzionale dei minimi quadrati pesati e regolarizzati:

$$
J_h(\boldsymbol{\theta}) = \frac{1}{2}\left\|W\bigl(\mathcal{G}_{h_{\mathrm{inv}}}(\boldsymbol{\theta}) - \mathbf{y}\bigr)\right\|_2^2 + \frac{\lambda}{2}\left\|L\bigl(\widetilde{\boldsymbol{\theta}} - \widetilde{\boldsymbol{\theta}}_0\bigr)\right\|_2^2, \qquad \widehat{\boldsymbol{\theta}}_h \in \operatorname*{arg\,min}_{\boldsymbol{\theta}\in\Theta} J_h(\boldsymbol{\theta}).
$$

### 3.2. Condizione necessaria di identificabilità da sole osservazioni di concentrazione

Se l'operatore osservativo $\mathcal{H}$ rileva esclusivamente la concentrazione $c$, i parametri meccanici $(\beta, E)$ influenzano le osservazioni solo indirettamente attraverso il campo di velocità di Darcy $\mathbf{q}_{\boldsymbol{\theta}} = -\mathbb{K}_h \nabla p_{\boldsymbol{\theta}}$. Come dimostrato nella **Proposizione di non-identificabilità a flusso nullo**, se $\mathbf{q}_{\boldsymbol{\theta}} \equiv \mathbf{0}$ al variare di $(E, \beta)$ (ad esempio in assenza di forzanti meccaniche o idrauliche e con dati iniziali meccanici nulli), le colonne del Jacobiano $J_{ij}^{\mathcal{G}}(\widetilde{\boldsymbol{\theta}}) = \partial \mathcal{G}_i / \partial \widetilde{\theta}_j$ relative a $\widetilde{\beta}$ ed $\widetilde{E}$ sono identicamente nulle. Questa proprietà strutturale è verificata automaticamente nel test `tests/test_05_identifiability_zero_q.py` e nello studio `experiments/02_run_identifiability_study.py`.

---

## 4. Architettura del Repository

```bash
Fractional-Poro-Viscoelastic/
├── CMakeLists.txt                        # Configurazione build C++20 (deal.II + Eigen3 + pybind11)
├── pyproject.toml                        # Configurazione pacchetto Python `reteco` e opzioni pytest
├── environment.yml                       # Specifica rigorosa dell'ambiente conda-forge
├── prepare.sh                            # Script automatico di bootstrap, build e provenance (macOS/Linux)
├── Makefile                              # Target operativi (setup, build, test, experiments, clean)
├── README.md                             # Questo documento
│
├── configs/                              # Configurazioni sperimentali versionate e immutabili
│   ├── scales_and_physics.yaml           # Scale (L0, T0=tau_c, E0, c_ref), fisica fissa e spazio Theta
│   ├── mms_benchmark.yaml                # Griglie spaziali e temporali per il test MMS accoppiato
│   ├── twin_experiment_synthetic.yaml    # Discretizzazione fine (target) vs standard (inversione)
│   └── sota_comparison.yaml              # Budget valutazioni FEM, seed e iperparametri ottimizzatori
│
├── cpp_core/                             # Nucleo agli Elementi Finiti in C++20 (deal.II)
│   ├── include/reteco/
│   │   ├── parameters.hpp                # Validazione di ParameterTheta e FixedPhysicsConfig
│   │   ├── time_grid.hpp                 # Partizioni temporali uniformi e graduate t_n = T*(n/N)^r
│   │   ├── caputo_l1_memory.hpp          # Formula L1 non uniforme per gli operatori di Caputo
│   │   ├── invariant_matrices.hpp        # Assemblaggio Taylor-Hood Q2/Q1 (Biot) e Q1 (Tracciante)
│   │   ├── initial_compatibility.hpp     # Calcolo e verifica di a_E(d0,v) - b(p0,div v) = <F(0),v>
│   │   ├── biot_visco_solver.hpp         # Solutore implicito del sistema poro-viscoelastico (d^n, p^n)
│   │   ├── tracer_transport_solver.hpp   # Solutore implicito del trasporto frazionario c^n
│   │   ├── energy_mass_monitor.hpp       # Diagnostica energetica E(t), D_beta(t) e massa M(t)
│   │   └── mms_manufactured.hpp          # Soluzioni esatte (d*, p*, c*) e forzanti analitiche MMS
│   ├── src/                              # Implementazioni C++20 corrispondenti
│   └── bindings/
│       └── pybind_module.cpp             # Binding Python `reteco._core` con guard runtime MPI/Kokkos
│
├── python/                               # Libreria scientifica Python (`reteco`)
│   └── reteco/
│       ├── __init__.py
│       ├── config_schema.py              # Validazione Pydantic e mappa di normalizzazione psi(theta)
│       ├── geometry/                     # Estrazione contorni NIfTI ( affine voxel -> mm -> L0 ) e Gmsh
│       ├── forward/                      # Interfaccia tipizzata al solutore C++ e generatore target
│       ├── observation/                  # Operatore osservativo H_h e generatore di rumore AWGN
│       ├── identifiability/              # Calcolo Jacobiano normalizzato, spettro SVD e sensitività
│       ├── inverse/                      # Funzionale J_h(theta), regolarizzazione, CPSO e runner SOTA
│       └── visualization/                # Generazione di grafici e tabelle LaTeX dai file JSON/CSV
│
├── tests/                                # Suite di verifica automatica (pytest)
│   ├── test_01_caputo_l1.py              # Verifica analitica dei pesi L1 su t^m e limite gamma -> 1
│   ├── test_02_initial_compatibility.py  # Verifica dell'equilibrio meccanico iniziale a t = 0
│   ├── test_03_energy_and_mass.py        # Verifica identità energetica discreta e conservazione M(t)=M(0)
│   ├── test_04_mms_coupled.py            # Verifica degli ordini di convergenza spaziali e temporali MMS
│   ├── test_05_identifiability_zero_q.py # Verifica Proposizione di non-identificabilità per q = 0
│   └── test_06_inverse_consistency.py    # Verifica contatore valutazioni FEM, bounds e determinismo seed
│
├── experiments/                          # Script di esecuzione end-to-end per la riproduzione dell'articolo
│   ├── 01_run_forward_verification.py    # Esegue la verifica completa del problema diretto
│   ├── 02_run_identifiability_study.py   # Esegue l'analisi del Jacobiano e dei valori singolari
│   ├── 03_run_twin_robustness.py         # Esegue i twin experiments multi-rumore (0%, 1%, 5%, 10%)
│   └── 04_run_sota_benchmark.py          # Esegue il benchmark comparativo a budget FEM fissato
│
└── data/                                 # Directory dati locali (esclusa da git tramite .gitignore)
    ├── raw/brats/                        # Volumi NIfTI (.nii.gz) del dataset BraTS (da scaricare)
    └── meshes/                           # Mesh Gmsh (.msh) generate dai profili anatomici o canonici
```

---

## 5. Acquisizione dei Dati MRI (Dataset BraTS) e Gestione `.gitignore`

Per ragioni di dimensione dei file binari e di conformità alle licenze d'uso dei dataset neuroradiologici pubblici, i volumi di risonanza magnetica in formato NIfTI (`.nii`, `.nii.gz`) sono esclusi dal controllo di versione tramite il file `.gitignore`.

### 5.1. Nota metodologica sull'uso delle immagini anatomiche
Come precisato nel manoscritto, una geometria estratta da un'immagine di risonanza magnetica definisce esclusivamente un **dominio computazionale bidimensionale non ideale** $\Omega_h$ (tramite conversione affine da coordinate voxel a millimetri fisici e successiva normalizzazione rispetto alla scala caratteristica $L_0$). Tale procedura **non** equivale a una serie temporale clinica dei campi $(\mathbf{d}, p, c)$ né costituisce una validazione clinica o fisiologica *in vivo*: tutti gli esperimenti di identificazione inversa condotti in questo repository sono **studi su dati sintetici (*twin experiments*)**.

### 5.2. Fonti ufficiali per il download del dataset BraTS
I volumi T1-pesati utilizzati per l'estrazione del contorno parenchimale provengono dal **Brain Tumor Segmentation (BraTS) Challenge Dataset**:

1. **Portale Ufficiale Synapse (RSNA-ASNR-MICCAI BraTS Challenge)**:
   * URL: [https://www.synapse.org/brats](https://www.synapse.org/brats)
2. **Center for Biomedical Image Computing and Analytics (CBICA — University of Pennsylvania)**:
   * URL: [https://www.med.upenn.edu/cbica/brats/](https://www.med.upenn.edu/cbica/brats/)
3. **The Cancer Imaging Archive (TCIA) — BraTS Collection**:
   * URL: [https://www.cancerimagingarchive.net/](https://www.cancerimagingarchive.net/)

### 5.3. Posizionamento dei file e conversione geometrica
Per riprodurre la generazione della mesh a partire da una scansione NIfTI:

1. Creare la directory locale dei dati grezzi (ignorata da Git):
   ```bash
   mkdir -p data/raw/brats data/meshes
   ```
2. Copiare il volume T1-pesato (es. `BraTS2021_00000_t1.nii.gz`) in `data/raw/brats/`.
3. Il modulo `python/reteco/geometry/mri_segmentation.py` legge la matrice affine $4 \times 4$ contenuta nell'header NIfTI (`nibabel`), converte le coordinate dei voxel in millimetri fisici e le scala rispetto a $L_0$ (definito in `configs/scales_and_physics.yaml`), generando tramite `gmsh_generator.py` la mesh conforme `.msh` con tag di bordo fisici distinti per $\Gamma_D$ (`boundary_id = 1`) e $\Gamma_N$ (`boundary_id = 2`).
4. **Modalità autonoma senza dataset esterno**: l'intera suite di verifica matematica (`tests/test_01_*.py` $\div$ `tests/test_06_*.py`) e gli script di verifica analitica e sintetica operano nativamente su domini canonici generati internamente da `deal.II` o su mesh poligonali pre-configurate, consentendo la verifica completa del codice anche prima del download dei volumi NIfTI da BraTS.

---

## 6. Requisiti di Sistema e Installazione Deterministica

L'infrastruttura è supportata su **macOS** (Apple Silicon `arm64` e Intel `x86_64`) e **Linux** (`x86_64`).

### 6.1. Prerequisiti software
* **Conda / Miniforge / Mamba**: per la creazione dell'ambiente isolato `./.conda_env` a partire da `environment.yml`.
* **Libreria `deal.II` ($\ge 9.5.0$, testato con `9.8.0`)**: può essere fornita tramite un'installazione locale preesistente (ad esempio compilata via `candi` in `~/dealii-candi/` o `/Applications/deal.II.app`) oppure installata automaticamente da `conda-forge`.
* **Compilatore C++20**: `clang++` (Xcode Command Line Tools su macOS) o `g++ >= 12` su Linux.

### 6.2. Procedura di installazione e compilazione automatica
Dalla directory radice del repository, eseguire i comandi seguenti:

```bash
# 1. Rendere eseguibile lo script di preparazione
chmod +x prepare.sh

# 2. Creare l'ambiente conda locale, configurare CMake e compilare il modulo C++20 `reteco._core`
./prepare.sh --build

# 3. Attivare l'ambiente deterministico nel terminale corrente
source ./activate_reteco.sh
```

**Cosa esegue `./prepare.sh --build`:**
1. Crea o aggiorna l'ambiente isolato in `./.conda_env` installando Python 3.11, `numpy`, `scipy`, `sympy`, `pydantic`, `pybind11`, `cmake`, `ninja`, `gmsh`, `nibabel` e `pytest`.
2. Individua l'installazione di `deal.II` (tramite variabile `DEAL_II_DIR`, installazione `candi` o ambiente conda) e verifica la compatibilità C++20.
3. Compila in modalità `Release` la libreria statica `libreteco_cpp_core.a` e l'estensione binaria Python `python/reteco/_core.cpython-311-*.so`.
4. Installa il pacchetto Python `reteco` in modalità *editable* (`pip install --no-deps --no-build-isolation -e .`).
5. Genera il certificato di provenienza dell'ambiente in `artifacts/environment_provenance.json` (contenente hash del commit Git, versione del compilatore, percorso di `deal.II`, architettura hardware e timestamp).

---

## 7. Suite di Verifica Automatica (`pytest`)

Prima di eseguire qualsiasi esperimento numerico per l'articolo, l'integrità matematica e numerica del solutore deve essere certificata tramite la suite di 6 test automatici.

Per eseguire l'intera suite con output dettagliato a schermo:

```bash
source ./activate_reteco.sh
pytest tests/ -v -s
```
*(In alternativa: `make test`)*.

### Dettaglio dei test eseguiti e criteri di superamento

| File di Test | Proprietà Matematica / Numerica Verificata | Criterio Quantitativo di Superamento (Asserzioni) |
| :--- | :--- | :--- |
| `tests/test_01_caputo_l1.py` | Formula di quadratura $L1$ per ${}^C D_t^\gamma u$ su griglie uniformi e graduate $t_n = T(n/N)^r$, e coerenza del limite classico $\gamma \uparrow 1$. | Errore al passo finale su $u(t)=t$ pari a zero macchina ($< 10^{-13}$); ordine di convergenza temporale empirico su $u(t)=t^2$ conforme a $2-\gamma$; convergenza a $\partial_t u$ per $\gamma = 0.99999$. |
| `tests/test_02_initial_compatibility.py` | Condizione di compatibilità meccanica iniziale $a_E(\mathbf{d}_0, \mathbf{v}) - b(p_0, \nabla\cdot\mathbf{v}) = \langle F(0), \mathbf{v}\rangle$ su $\mathbf{v} \in V$. | Norma $L^2$ discreta del residuo sulle componenti libere di $V$ inferiore a $10^{-11}$ sia per lo stato nullo compatibile sia per una pressione iniziale $p_0(\mathbf{x}) \neq 0$ assegnata. |
| `tests/test_03_energy_and_mass.py` | Positività della dissipazione viscoelastica $\mathcal{D}_\beta(t) \ge 0$, bilancio energetico discreto (Proposizione 1) e conservazione della massa $M(t)=M(0)$. | Decadimento monotono $\mathcal{E}(t_n) + \mathcal{D}_\beta(t_n) \le \mathcal{E}(0)$ nel caso omogeneo; identità energetica puntuale discreta (inclusa dissipazione numerica di Eulero implicito) verificata a $< 10^{-10}$; variazione relativa di massa $\|M(t_n)-M(0)\| / M(0) < 10^{-12}$. |
| `tests/test_04_mms_coupled.py` | Metodo delle Soluzioni Manufatte (MMS) sul sistema accoppiato completo $(\mathbf{d}^*, p^*, c^*)$ in $\Omega = (0,1)^2$. | Riduzione monotona degli errori $L^2$ e $H^1$ per tutti e tre i campi al raffinamento spaziale ($N \in \{8, 16, 32\}$) e temporale ($M \in \{16, 32, 64\}$) con verifica esplicita degli ordini empirici di convergenza. |
| `tests/test_05_identifiability_zero_q.py` | Proposizione di non-identificabilità di $(E, \beta)$ da osservazioni della sola concentrazione $c$ quando $\mathbf{q}_{\boldsymbol{\theta}} \equiv \mathbf{0}$. | Per $\mathbf{q} \equiv \mathbf{0}$: colonne del Jacobiano $\partial\mathcal{G}/\partial\widetilde{\beta}$ e $\partial\mathcal{G}/\partial\widetilde{E}$ nulle ($< 10^{-12}$) e rango numerico $\le 2$. Con carico idraulico/meccanico attivo ($\mathbf{q} \neq \mathbf{0}$): colonne non nulle e rango numerico pari a $4$. |
| `tests/test_06_inverse_consistency.py` | Coerenza del funzionale obiettivo $J_h(\boldsymbol{\theta})$, invertibilità di $\widetilde{\boldsymbol{\theta}}=\psi(\boldsymbol{\theta})$, tracciamento del budget FEM e riproducibilità col seed. | Errore di inversione $\|\psi^{-1}(\psi(\boldsymbol{\theta})) - \boldsymbol{\theta}\| < 10^{-14}$; incremento esatto del contatore `fem_evaluation_count`; coincidenza bit-a-bit di due esecuzioni indipendenti di CPSO a parità di `seed`. |

---

## 8. Pipeline di Esecuzione degli Esperimenti e Output Generati

Tutti gli script sperimentali salvano i propri risultati in sottocartelle strutturate dentro `artifacts/`, accompagnando ogni esecuzione con una copia della configurazione YAML utilizzata, i semi casuali e i file dati grezzi (`.json` e `.csv`) da cui vengono generate le tabelle LaTeX e le figure dell'articolo.

### 8.1. Esperimento 1 — Verifica Numerica del Problema Diretto
Esegue la verifica dei pesi $L1$, lo studio di convergenza spaziale e temporale MMS per $(\mathbf{d}^*, p^*, c^*)$ e la diagnostica di conservazione energetica e di massa.

* **Comando di esecuzione**:
  ```bash
  python experiments/01_run_forward_verification.py
  ```
* **Output generati in `artifacts/exp01_forward_verification/`**:
  * `mms_spatial_convergence.csv` e `mms_temporal_convergence.csv`: errori $L^2(\Omega)$ e $H^1(\Omega)$ e tassi empirici di convergenza (EOC) per $\mathbf{d}_h$, $p_h$, $c_h$.
  * `energy_and_mass_history.csv`: evoluzione temporale di $\mathcal{E}(t_n)$, $\mathcal{D}_\beta(t_n)$, dissipazione idraulica $\int_0^{t_n} a_h(p, p)\,\mathrm{d}s$, dissipazione numerica di passo e massa totale $M(t_n)$.
  * `table_mms_convergence.tex`: tabella LaTeX formattata pronta per l'inclusione nel manoscritto.
  * `fig_mms_convergence.pdf` e `fig_energy_mass_balance.pdf`: grafici di convergenza e bilanci strutturali.
  * `run_manifest.json`: metadati completi dell'esecuzione.

### 8.2. Esperimento 2 — Studio di Identificabilità Locale e Spettro SVD
Calcola la matrice Jacobiana normalizzata $J^{\mathcal{G}}(\widetilde{\boldsymbol{\theta}}^\dagger)$ tramite differenze finite adattive centrate, ne analizza la decomposizione ai valori singolari $\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge \sigma_4$ e confronta la configurazione a flusso nullo ($\mathbf{q} \equiv \mathbf{0}$) con i protocolli osservativi a flusso di Darcy attivo.

* **Comando di esecuzione**:
  ```bash
  python experiments/02_run_identifiability_study.py
  ```
* **Output generati in `artifacts/exp02_identifiability/`**:
  * `svd_spectrum_comparison.json`: valori singolari $\sigma_k$, numero di condizionamento $\kappa(J^{\mathcal{G}}) = \sigma_1 / \sigma_4$, rango numerico effettivo e vettori singolari destri $V_k$ nello spazio $(\widetilde{\alpha}, \widetilde{\beta}, \widetilde{D}, \widetilde{E})$.
  * `sensitivity_profiles.csv`: norme temporali delle colonne di sensibilità $\|\partial \mathcal{G}_h(t_n) / \partial \widetilde{\theta}_j\|_2$.
  * `table_identifiability_svd.tex`: tabella LaTeX dei valori singolari e degli indici di collinearità.
  * `fig_svd_spectrum_and_sensitivities.pdf`: figura comparativa dello spettro SVD tra regime a flusso nullo e regime forzato.

### 8.3. Esperimento 3 — *Twin Experiments* e Robustezza al Rumore Osservativo
Genera le osservazioni sintetiche $\mathbf{y} = \mathcal{G}_{h_{\mathrm{tar}}}(\boldsymbol{\theta}^\dagger) + \boldsymbol{\epsilon}$ su una discretizzazione spaziale e temporale raffinata ($h_{\mathrm{tar}} < h_{\mathrm{inv}}$, $\Delta t_{\mathrm{tar}} < \Delta t_{\mathrm{inv}}$), quantifica preliminarmente la discrepanza di discretizzazione $\|\mathcal{G}_{h_{\mathrm{tar}}}(\boldsymbol{\theta}^\dagger) - \mathcal{G}_{h_{\mathrm{inv}}}(\boldsymbol{\theta}^\dagger)\|_2$ ed esegue l'identificazione inversa su repliche indipendenti ai livelli di rumore additivo gaussiano $\sigma_{\mathrm{noise}} \in \{0\%, 1\%, 5\%, 10\%\}$.

* **Comando di esecuzione**:
  ```bash
  python experiments/03_run_twin_robustness.py
  ```
* **Output generati in `artifacts/exp03_twin_robustness/`**:
  * `discretization_discrepancy_audit.json`: misura dell'errore di discretizzazione tra mesh target e mesh di inversione prima dell'aggiunta del rumore.
  * `twin_robustness_replicates.csv`: stime $\widehat{\boldsymbol{\theta}}_h$, errori relativi percentuali su $(\alpha, \beta, D, E)$, valore finale di $J_h(\widehat{\boldsymbol{\theta}}_h)$ e seed di ogni replica.
  * `twin_robustness_summary.json` e `table_twin_robustness.tex`: media, deviazione standard, mediana e intervallo interquartile (IQR) degli errori di ricostruzione per ciascun livello di rumore.
  * `fig_twin_parameter_recovery.pdf`: distribuzione delle stime e profili di concentrazione ricostruiti rispetto al target sintetico pulito e perturbato.

### 8.4. Esperimento 4 — Benchmark Comparativo degli Ottimizzatori a Budget Fissato
Confronta l'algoritmo *Continuous Particle Swarm Optimization* (`CPSO`) con ottimizzatori deterministici e stocastici di riferimento (`L-BFGS-B`, `Powell`, `Nelder-Mead`, `Differential-Evolution`) imponendo **lo stesso modello diretto**, **lo stesso operatore osservativo $\mathcal{H}$**, **lo stesso funzionale obiettivo $J_h$**, **gli stessi limiti $\Theta$** e **lo stesso budget massimo di valutazioni del solutore FEM** (`max_fem_evaluations` specificato in `configs/sota_comparison.yaml`).

* **Comando di esecuzione**:
  ```bash
  python experiments/04_run_sota_benchmark.py
  ```
* **Output generati in `artifacts/exp04_sota_benchmark/`**:
  * `sota_benchmark_runs.csv`: registro completo di ogni run, ottimizzatore, seed iniziale, numero esatto di chiamate FEM effettuate, tempo di calcolo (wall-clock time), valore ottimo $J_h(\widehat{\boldsymbol{\theta}}_h)$ ed errori sui singoli parametri.
  * `sota_convergence_histories.json`: curve di decadimento del miglior valore di $J_h$ in funzione del numero di valutazioni del problema diretto.
  * `table_sota_comparison.tex`: tabella comparativa LaTeX generata direttamente dai file CSV certificati.
  * `fig_sota_convergence_budget.pdf`: curve di convergenza rispetto al budget di valutazioni FEM.

Per eseguire l'intera sequenza di verifica e tutti i quattro esperimenti con un unico comando:

```bash
make all-experiments
```

---

## 9. Tracciabilità, Criteri Anti-Inverse-Crime e Limiti Metodologici

In conformità con il protocollo scientifico adottato nel manoscritto, l'interpretazione dei risultati prodotti da questo codice rispetta i seguenti vincoli:

1. **Separazione tra stime regolari ed esistenza debole**: le verifiche energetiche e di stabilità implementate in `energy_mass_monitor.cpp` controllano la coerenza discreta rispetto all'identità energetica dimostrata per soluzioni regolari; esse non sostituiscono una dimostrazione teorica di esistenza debole in spazi minimi per il sistema continuo pienamente accoppiato.
2. **Prevenzione dell'*inverse crime***: in `configs/twin_experiment_synthetic.yaml`, la mesh e il passo temporale usati per generare il dato sintetico $\mathbf{y}$ sono strettamente più fini rispetto a quelli usati dall'operatore di inversione $\mathcal{G}_{h_{\mathrm{inv}}}$. Il codice rifiuta l'esecuzione del twin experiment se $h_{\mathrm{tar}} = h_{\mathrm{inv}}$ e $\Delta t_{\mathrm{tar}} = \Delta t_{\mathrm{inv}}$, a meno che non venga esplicitamente richiesto un test diagnostico a discretizzazione coincidente.
3. **Parità di condizioni nel confronto tra ottimizzatori**: tutti gli algoritmi in `python/reteco/inverse/sota_runners.py` invocano la medesima istanza di `ObjectiveFunctional`, la quale incrementa un contatore interno a ogni chiamata di `solve_trajectory` e interrompe la ricerca al raggiungimento esatto di `max_fem_evaluations`.
4. **Distinzione tra dati sintetici e validazione clinica**: le ricostruzioni ottenute su domini estratti da immagini BraTS misurano esclusivamente le prestazioni numeriche dell'identificazione parametrica nel protocollo sintetico definito, e non vengono mai etichettate nei grafici o nelle tabelle come misure cliniche o parametri fisiologici reali.

---

## 10. Citazione e Riferimenti Bibliografici

Se utilizzi questo codice o la formulazione del modello nel tuo lavoro di ricerca, sei pregato di citare il manoscritto di riferimento:

```bibtex
@article{Iacuzzo2026,
  author  = {Iacuzzo, Giovanni Giuseppe},
  title   = {Parameter Identification in Fractional Poro-Viscoelastic Media: Application to Brain White Matter},
  journal = {},
  year    = {},
  note    = {}
}
```

### Principali riferimenti teorici e metodologici
* **M. A. Biot (1941)**, *General theory of three-dimensional consolidation*, Journal of Applied Physics, 12(2), 155–164.
* **I. Podlubny (1999)**, *Fractional Differential Equations*, Academic Press, San Diego.
* **M. Caponi, F. Sapio (2021)**, *A dynamic model for viscoelastic materials with fractional memory*, Evolution Equations and Control Theory.
* **L. Bociu, B. Muha, J. T. Webster (2023)**, *Weak solutions in nonlinear poroelasticity with viscoelasticity*, Nonlinear Analysis: Real World Applications.
* **N. Kopteva (2019)**, *Error analysis of the L1 method on graded and uniform meshes for a fractional-derivative problem in two and three dimensions*, Mathematics of Computation, 88(319), 2135–2155.
* **A. Cardone, M. Frasca-Caccia (2022)**, *Numerical conservation laws of time fractional diffusion PDEs*, Fractional Calculus and Applied Analysis, 25, 1459–1483.
* **B. H. Menze et al. (2015)**, *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)*, IEEE Transactions on Medical Imaging, 34(10), 1993–2024.