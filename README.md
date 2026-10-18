# Modello Poro-Viscoelastico Frazionario

Questo repository contiene l'implementazione scientifica di riferimento (**C++20 / `deal.II` 9.8 + Python 3.11**) e il manoscritto LaTeX associati allo studio del problema diretto, della sensibilità multifisica su geometrie anatomiche derivate da risonanza magnetica (BraTS) e del problema inverso di identificazione parametrica per un mezzo poro-viscoelastico frazionario di tipo Biot accoppiato unidirezionalmente al trasporto sub-diffusivo di un tracciante passivo.

L'infrastruttura software è progettata secondo cinque principi metodologici rigorosi, formalizzati nel documento di specifica `contratto_matematico.md`:

1. **Coerenza tra formulazione continua, adimensionalizzazione e discretizzazione**: separazione esplicita tra il sottosistema meccanico-idraulico quasi-statico con memoria frazionaria di Kelvin–Voigt ($\beta, E$) e l'equazione di advezione-diffusione frazionaria di Caputo per il tracciante passivo ($\alpha, D$), con scala temporale caratteristica vincolata a $T_0 = \tau_c$.
2. **Verifica numerica indipendente del problema diretto**: test automatici dei pesi temporali $L1$ su griglie uniformi e graduate, controllo della condizione di compatibilità meccanica iniziale a $t=0$, verifica dell'identità energetica discreta passo-passo (inclusa la dissipazione numerica esatta dello schema di Eulero implicito) e della conservazione della massa $M(t)=M(0)$, e verifica degli ordini di convergenza spaziali e temporali tramite il Metodo delle Soluzioni Manufatte (MMS).
3. **Integrazione rigorosa di domini anatomici 2D da neuroimaging multimodale (BraTS)**: pipeline automatizzata di estrazione di sezioni assiali da volumi NIfTI, trasformazione affine dal reticolo voxel alle coordinate fisiche millimetriche e adimensionali ($L_0$), generazione di mesh quadrilaterali conformi in Gmsh e importazione diretta in `deal.II` con verifica algebrica delle matrici di massa e dei gradi di libertà.
4. **Analisi multi-livello di identificabilità preliminare all'inversione**: studio strutturale (verifica della Proposizione di non-identificabilità a flusso di Darcy nullo $\mathbf{q}_{\boldsymbol{\theta}} \equiv \mathbf{0}$), analisi locale tramite matrice Jacobiana normalizzata, spettro dei valori singolari (SVD) e matrice di correlazione locale, e analisi globale/pratica tramite sezioni *One-At-a-Time* (OAT) e profili di funzione obiettivo ottimizzati (*Profile Likelihood*).
5. **Protocollo sperimentale inverso riproducibile ed esente da *inverse crime***: separazione stretta tra la discretizzazione di generazione dei dati sintetici ($\mathcal{G}_{h_{\mathrm{tar}}}$) e la discretizzazione impiegata nell'inversione ($\mathcal{G}_{h_{\mathrm{inv}}}$), tracciamento immutabile del budget di valutazioni agli elementi finiti (FEM) e registrazione completa dei semi pseudocasuali e della provenienza dell'ambiente di calcolo (`artifacts/environment_provenance.json`).

---

## Indice

1. [Formulazione Matematica di Riferimento](#1-formulazione-matematica-di-riferimento)
2. [Discretizzazione Numerica e Integrazione Mesh Anatomiche in `deal.II`](#2-discretizzazione-numerica-e-integrazione-mesh-anatomiche-in-dealii)
3. [Problema Inverso e Analisi Multi-Livello di Identificabilità](#3-problema-inverso-e-analisi-multi-livello-di-identificabilit%C3%A0)
4. [Architettura Completa del Repository](#4-architettura-completa-del-repository)
5. [Acquisizione dei Dati MRI (Dataset BraTS), Pipeline Anatomica e `.gitignore`](#5-acquisizione-dei-dati-mri-dataset-brats-pipeline-anatomica-e-gitignore)
6. [Requisiti di Sistema e Installazione Deterministica](#6-requisiti-di-sistema-e-installazione-deterministica)
7. [Suite di Verifica Automatica (`pytest` — 10 Moduli di Test)](#7-suite-di-verifica-automatica-pytest--10-moduli-di-test)
8. [Pipeline di Esecuzione degli Esperimenti e Figure del Manoscritto](#8-pipeline-di-esecuzione-degli-esperimenti-e-figure-del-manoscritto)
9. [Struttura e Compilazione del Manoscritto (`paper/`)](#9-struttura-e-compilazione-del-manoscritto-paper)
10. [Tracciabilità, Criteri Anti-Inverse-Crime e Limiti Metodologici](#10-tracciabilit%C3%A0-criteri-anti-inverse-crime-e-limiti-metodologici)
11. [Citazione e Riferimenti Bibliografici](#11-citazione-e-riferimenti-bibliografici)

---

## 1. Formulazione Matematica di Riferimento

### 1.1. Variabili di stato e operatori frazionari

Sia $\Omega \subset \mathbb{R}^m$, con $m \in \{2, 3\}$, un dominio limitato con bordo Lipschitz $\partial\Omega = \overline{\Gamma_D} \cup \overline{\Gamma_N}$ e sia $t \in (0, T]$. Il modello descrive l'evoluzione spazio-temporale di:

* $\mathbf{d}(\mathbf{x}, t)$: spostamento della matrice solida porosa;
* $p(\mathbf{x}, t)$: pressione interstiziale del fluido nei pori;
* $\mathbf{q}(\mathbf{x}, t)$: flusso volumetrico specifico di Darcy;
* $c(\mathbf{x}, t)$: concentrazione del tracciante passivo trasportato nel mezzo.

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

Con questa convenzione il coefficiente $D_{\mathrm{phys}}$ conserva l'unità fisica classica $L^2/T$, mentre la dipendenza dall'ordine $\alpha$ risiede nell'operatore di accumulo ${}^C D_t^\alpha c$ e nel fattore di scala $\tau_c^{\alpha-1}$ prima dell'adimensionalizzazione. Il vincolo $T_0 = \tau_c$ è verificato a runtime sia in C++ (`reteco::FixedPhysicsConfig::validate`) sia in Python (`CharacteristicScales`).

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

dove $\mathbb{C}_0(\nu)$ è il tensore costitutivo elastico isotropo normalizzato (in ipotesi di deformazione piana nel caso bidimensionale, con $-1 < \nu < 1/2$), $b \in (0, 1]$ il coefficiente di Biot, $S_p > 0$ il coefficiente di immagazzinamento specifico e $\phi_0 \in (0, 1]$ la porosità di riferimento. L'accoppiamento è **unidirezionale**: il sottosistema poro-viscoelastico $(\mathbf{d}, p)$ determina il flusso di Darcy $\mathbf{q}$, il quale trasporta il tracciante $c$ senza retroazione della concentrazione sulle equazioni meccaniche o idrauliche.

Il vettore dei quattro parametri oggetto di identificazione è:

$$
\boldsymbol{\theta} = (\alpha, \beta, D, E)^{\mathsf{T}} \in \Theta = [\alpha_-, \alpha_+] \times [\beta_-, \beta_+] \times [D_-, D_+] \times [E_-, E_+],
$$

con $0 < \alpha_- < \alpha_+ < 1$, $0 < \beta_- < \beta_+ < 1$, $D_- > 0$ ed $E_- > 0$.

### 1.3. Condizioni al contorno, compatibilità iniziale e bilanci strutturali

* **Condizioni al contorno**:

$$
\mathbf{d} = \mathbf{0} \text{ su } \Gamma_D, \qquad \boldsymbol{\sigma}\mathbf{n} = \bar{\mathbf{t}} \text{ su } \Gamma_N, \qquad \mathbf{q}\cdot\mathbf{n} = 0 \text{ su } \partial\Omega, \qquad \mathbf{F}_c\cdot\mathbf{n} := (c\mathbf{q} - \phi_0 D\nabla c)\cdot\mathbf{n} = 0 \text{ su } \partial\Omega.
$$

* **Condizione di compatibilità meccanica iniziale**: per soluzioni regolari tali che ${}^C D_t^\beta \boldsymbol{\varepsilon}(\mathbf{d}(t)) \to \mathbf{0}$ per $t \downarrow 0$, lo stato iniziale $(\mathbf{d}_0, p_0)$ deve soddisfare l'equilibrio elastico-porale a $t = 0$:

$$
a_E(\mathbf{d}_0, \mathbf{v}) - b(p_0, \nabla\cdot\mathbf{v})_\Omega = \langle F(0), \mathbf{v}\rangle \quad \forall \mathbf{v} \in V.
$$

* **Identità energetica continua e discreta**: posto $\mathcal{E}(t) = \frac{1}{2}a_E(\mathbf{d}(t), \mathbf{d}(t)) + \frac{S_p}{2}\Vert p(t)\Vert_{L^2(\Omega)}^2$ e $\mathcal{D}_\beta(t) = \int_0^t a_v({}^C D_s^\beta \mathbf{d}(s), \partial_s \mathbf{d}(s))\,\mathrm{d}s \ge 0$, vale l'identità continua:

$$
\mathcal{E}(t) + \mathcal{D}_\beta(t) = \mathcal{E}(0) - \int_0^t a_h(p(s), p(s))\,\mathrm{d}s + \int_0^t \langle F(s), \partial_s\mathbf{d}(s)\rangle\,\mathrm{d}s + \int_0^t \langle s_f(s), p(s)\rangle\,\mathrm{d}s.
$$

Nello schema discreto implicito, detta $R_n = \mathcal{E}(t_n) + \mathcal{D}_{\beta,n} + \mathcal{D}_{h,n} - \mathcal{E}(0) - W_{\mathrm{ext},n}$ la discrepanza energetica cumulativa al passo $n$ e $D_{\mathrm{num},n} = \frac{E}{2}\Vert \mathbf{d}^n - \mathbf{d}^{n-1}\Vert_{K_0}^2 + \frac{S_p}{2}\Vert p^n - p^{n-1}\Vert_{M_p}^2 \ge 0$ la dissipazione numerica introdotta dal passo di Eulero implicito, la classe `reteco::EnergyMassMonitor` verifica a ogni passo l'identità algebrica esatta:

$$
(R_n - R_{n-1}) + D_{\mathrm{num},n} = 0.
$$

* **Conservazione della massa integrata del tracciante**: per $s_c = 0$ e flusso totale nullo al bordo ($\mathbf{F}_c \cdot \mathbf{n} = 0$), la massa spaziale integrata $M(t) = \int_\Omega c(\mathbf{x}, t)\,\mathrm{d}\mathbf{x}$ soddisfa ${}^C D_t^\alpha M(t) = 0$, da cui $M(t) = M(0)$ per ogni $t \in [0, T]$.

---

## 2. Discretizzazione Numerica e Integrazione Mesh Anatomiche in `deal.II`

### 2.1. Schema temporale $L1$ su griglie non uniformi

Sia $0 = t_0 < t_1 < \cdots < t_N = T$ una partizione temporale generale (ad esempio una griglia graduata $t_n = T(n/N)^r$ con esponente di graduazione $r \ge 1$) e passi $\Delta t_k = t_k - t_{k-1}$. L'operatore di Caputo di ordine $\gamma \in \{\alpha, \beta\}$ è approssimato mediante la formula $L1$ non uniforme:

$$
{}^C D_t^\gamma u(t_n) \approx \frac{1}{\Gamma(2-\gamma)} \sum_{k=1}^{n} \frac{u^k - u^{k-1}}{\Delta t_k} \left[(t_n - t_{k-1})^{1-\gamma} - (t_n - t_k)^{1-\gamma}\right].
$$

La classe C++ `reteco::CaputoL1Memory` calcola i pesi esatti per ciascun passo $t_n$ senza mai riutilizzare pesi uniformi quando $r > 1$, separando il coefficiente implicito corrente $d_{n,n}^{(\gamma)}$ dal termine di memoria storica dipendente da $\{u^0, \dots, u^{n-1}\}$.

### 2.2. Elementi finiti misti e supporto per mesh quadrilaterali anatomiche

* **Sottosistema poro-viscoelastico** $(\mathbf{d}_h, p_h)$: discretizzato mediante coppia stabile di tipo Taylor–Hood su quadrilateri, con elementi biquadratici $[Q_2]^2$ (`FE_Q<2>(2)`) per lo spostamento $\mathbf{d}_h$ ed elementi bilineari $Q_1$ (`FE_Q<2>(1)`) per la pressione $p_h$.
* **Trasporto del tracciante** $c_h$: formulazione di Galerkin continua su spazio $Q_1$ (`FE_Q<2>(1)`), con assemblaggio esatto della matrice di trasporto advettivo guidata dal campo di velocità di Darcy $\mathbf{q}_h = -\mathbb{K}_h \nabla p_h$ e opzione di stabilizzazione SUPG dichiarata.
* **Doppio costruttore geometrico in `reteco::InvariantFEMMatrices2D`**:
  1. *Dominio canonico*: generazione interna tramite `dealii::GridGenerator::subdivided_hyper_cube` su $\Omega = (0,1)^2$ per i benchmark MMS, i test strutturali e gli esperimenti sintetici su quadrato unitario.
  2. *Dominio anatomico importato*: lettura diretta di mesh quadrilaterali conformi Gmsh (`.msh`) tramite `dealii::GridIn<2>`, preservando i `boundary_id` fisici ($\Gamma_D$ e $\Gamma_N$) e i `material_id` delle sottoregioni anatomiche, con verifica automatica di coerenza (`verify_dof_and_mesh_consistency`) della partizione dell'unità delle matrici di massa $\mathbf{M}_d$, $\mathbf{M}_p$, $\mathbf{M}_c$.

---

## 3. Problema Inverso e Analisi Multi-Livello di Identificabilità

### 3.1. Operatore osservativo e funzionale obiettivo

Detta $X_{\boldsymbol{\theta}, h} = (\mathbf{d}_{\boldsymbol{\theta}, h}, p_{\boldsymbol{\theta}, h}, c_{\boldsymbol{\theta}, h})$ la soluzione discreta del problema diretto e $\mathcal{H}$ l'operatore osservativo che seleziona i campi, i sensori spaziali e gli istanti di campionamento, la mappa osservativa discreta è $\mathcal{G}_h(\boldsymbol{\theta}) = \mathcal{H}(X_{\boldsymbol{\theta}, h})$.

Nelle coordinate normalizzate $\widetilde{\boldsymbol{\theta}} = \psi(\boldsymbol{\theta}) \in [0, 1]^4$:

$$
\widetilde{\alpha} = \frac{\alpha - \alpha_-}{\alpha_+ - \alpha_-}, \quad \widetilde{\beta} = \frac{\beta - \beta_-}{\beta_+ - \beta_-}, \quad \widetilde{D} = \frac{\log D - \log D_-}{\log D_+ - \log D_-}, \quad \widetilde{E} = \frac{\log E - \log E_-}{\log E_+ - \log E_-},
$$

la stima inversa minimizza il funzionale dei minimi quadrati pesati e regolarizzati:

$$
J_h(\boldsymbol{\theta}) = \frac{1}{2}\left\Vert W\bigl(\mathcal{G}_{h_{\mathrm{inv}}}(\boldsymbol{\theta}) - \mathbf{y}\bigr)\right\Vert_2^2 + \frac{\lambda}{2}\left\Vert L\bigl(\widetilde{\boldsymbol{\theta}} - \widetilde{\boldsymbol{\theta}}_0\bigr)\right\Vert_2^2, \qquad \widehat{\boldsymbol{\theta}}_h \in \operatorname*{arg\,min}_{\boldsymbol{\theta}\in\Theta} J_h(\boldsymbol{\theta}).
$$

### 3.2. Diagnostica di identificabilità strutturale, locale e pratica

Il pacchetto `python/reteco/identifiability/` implementa quattro livelli complementari di analisi:

1. **Verifica strutturale a flusso nullo (Proposizione 2)**: se vengono osservati soltanto dati di concentrazione $c$ e il flusso di Darcy soddisfa $\mathbf{q}_{\boldsymbol{\theta}} \equiv \mathbf{0}$ al variare di $(E, \beta)$, allora le colonne del Jacobiano rispetto a $\widetilde{\beta}$ ed $\widetilde{E}$ sono identicamente nulle ($\partial\mathcal{G}/\partial\widetilde{\beta} = \mathbf{0}$, $\partial\mathcal{G}/\partial\widetilde{E} = \mathbf{0}$).
2. **Spettro dei valori singolari (SVD) del Jacobiano normalizzato** $J^{\mathcal{G}}(\widetilde{\boldsymbol{\theta}}^\dagger)$: quantifica il rango numerico effettivo, il numero di condizionamento $\kappa(J^{\mathcal{G}}) = \sigma_1 / \sigma_4$ e il divario di scala tra i parametri di trasporto diretto ($\widetilde{\alpha}, \widetilde{D}$) e i parametri idro-meccanici ($\widetilde{\beta}, \widetilde{E}$) al variare del protocollo osservativo (sola concentrazione $c$, multimodale $(c, p)$, multimodale completo $(c, p, \mathbf{d})$).
3. **Matrice di correlazione locale approssimata**: derivata dall'inversa della matrice di informazione di Fisher $F = (J^{\mathcal{G}})^{\mathsf{T}} W^{\mathsf{T}} W J^{\mathcal{G}}$, individua le coppie parametriche affette da collinearità locale.
4. **Sezioni *One-At-a-Time* (OAT) e *Profile Likelihood* (`profile_likelihood.py`)**: confronta le sezioni 1D a parametri bloccati $s_j(\widetilde{\theta}_j) = J_h(\widetilde{\theta}_j, \widetilde{\boldsymbol{\theta}}_{-j}^\dagger)$ con i profili di funzione obiettivo ri-ottimizzati sui parametri di disturbo:

$$
\widehat{J}_j(\widetilde{\theta}_j) = \min_{\widetilde{\boldsymbol{\theta}}_{-j} \in [0,1]^3} J_h(\widetilde{\theta}_j, \widetilde{\boldsymbol{\theta}}_{-j}),
$$

rivelando eventuali direzioni di compensazione non lineare e *sloppiness* pratica.

---

## 4. Architettura Completa del Repository

```bash
Fractional-Poro-Viscoelastic/
├── CMakeLists.txt                        # Configurazione build C++20 (deal.II 9.8 + Eigen3 + pybind11)
├── pyproject.toml                        # Configurazione pacchetto Python `reteco` e opzioni pytest
├── environment.yml                       # Specifica rigorosa dell'ambiente conda-forge
├── prepare.sh                            # Bootstrap automatico dell'ambiente, build C++20 e provenance
├── activate_reteco.sh                    # Script generato da prepare.sh per attivare l'ambiente deterministico
├── Makefile                              # Target operativi (setup, build, test, esperimenti, clean)
├── contratto_matematico.md               # Specifica formale delle equazioni, scale e gate di verifica
├── LICENSE                               # Licenza d'uso del software
├── README.md                             # Questo documento
│
├── configs/                              # Configurazioni sperimentali versionate e immutabili (YAML)
│   ├── scales_and_physics.yaml           # Scale (L0, T0=tau_c, E0, c_ref), fisica fissa e spazio Theta
│   ├── mms_benchmark.yaml                # Griglie spaziali e temporali per il test MMS accoppiato
│   ├── brats_anatomical_forward.yaml     # Parametri di estrazione ROI BraTS, meshing Gmsh e studio anatomico
│   ├── twin_experiment_synthetic.yaml    # Discretizzazione fine (target) vs standard (inversione)
│   └── sota_comparison.yaml              # Budget valutazioni FEM, seed e iperparametri ottimizzatori
│
├── cpp_core/                             # Nucleo agli Elementi Finiti in C++20 (deal.II)
│   ├── include/reteco/
│   │   ├── parameters.hpp                # Validazione rigorosa di ParameterTheta e FixedPhysicsConfig
│   │   ├── time_grid.hpp                 # Partizioni temporali uniformi e graduate t_n = T*(n/N)^r
│   │   ├── caputo_l1_memory.hpp          # Formula L1 non uniforme per gli operatori di Caputo
│   │   ├── invariant_matrices.hpp        # Assemblaggio Taylor-Hood Q2/Q1 + Q1 su ipercubo o mesh Gmsh (.msh)
│   │   ├── initial_compatibility.hpp     # Calcolo e verifica di a_E(d0,v) - b(p0,div v) = <F(0),v>
│   │   ├── biot_visco_solver.hpp         # Solutore implicito del sistema poro-viscoelastico (d^n, p^n)
│   │   ├── tracer_transport_solver.hpp   # Solutore implicito del trasporto frazionario c^n (con opzione SUPG)
│   │   ├── energy_mass_monitor.hpp       # Diagnostica E(t), D_beta(t), dissipazione numerica e massa M(t)
│   │   └── mms_manufactured.hpp          # Soluzioni esatte (d*, p*, c*) e forzanti analitiche MMS
│   ├── src/                              # Implementazioni C++20 corrispondenti (.cpp)
│   └── bindings/
│       └── pybind_module.cpp             # Binding Python `reteco._core` con DealIIRuntimeGuard (MPI/Kokkos)
│
├── python/                               # Framework Scientifico e Problema Inverso in Python (`reteco`)
│   └── reteco/
│       ├── __init__.py
│       ├── config_schema.py              # Validazione Pydantic e mappa di normalizzazione psi(theta)
│       ├── anatomy/                      # Pipeline anatomica multimodale BraTS e controllo qualità mesh
│       │   ├── __init__.py
│       │   └── brats_geometry.py         # Estrazione slice NIfTI, maschere tissutali e meshing quadrilaterale
│       ├── geometry/                     # Segmentazione contorni MRI e generazione mesh Gmsh 2D
│       │   ├── mri_segmentation.py
│       │   └── gmsh_generator.py
│       ├── forward/                      # Interfaccia tipizzata al solutore C++ e generatore target sintetico
│       │   ├── wrapper.py
│       │   └── target_generator.py
│       ├── observation/                  # Operatore osservativo H_h e generatore di rumore AWGN
│       │   ├── operator.py
│       │   └── noise_model.py
│       ├── identifiability/              # Analisi di identificabilità strutturale, locale e pratica
│       │   ├── __init__.py
│       │   ├── sensitivity_jacobian.py   # Calcolo Jacobiano normalizzato con differenze finite adattive
│       │   ├── svd_conditioning.py       # Decomposizione SVD, rango numerico e correlazione locale
│       │   └── profile_likelihood.py     # Sezioni OAT 1D e profili obiettivo ri-ottimizzati
│       ├── inverse/                      # Ottimizzazione inversa e benchmark SOTA a budget fissato
│       │   ├── objective_functional.py   # Funzionale J_h(theta) condiviso con contatore FEM immutabile
│       │   ├── regularization.py         # Regolarizzazione di Tikhonov e principio di Morozov
│       │   ├── cpso.py                   # Continuous Particle Swarm Optimization (CPSO) deterministico
│       │   └── sota_runners.py           # Runner equi per CPSO, L-BFGS-B, Powell, Nelder-Mead, DE
│       └── visualization/                # Generazione automatica delle figure PDF per il manoscritto
│           ├── verification_plots.py
│           ├── anatomical_forward_plots.py
│           ├── identifiability_plots.py
│           └── benchmark_plots.py
│
├── scripts/                              # Script operativi per la pipeline anatomica BraTS
│   ├── run_brats_anatomical_pipeline.py  # Estrae la geometria BraTS, genera le mesh .msh e valida i DoF
│   └── run_anatomical_forward_simulation.py # Esegue una simulazione diretta completa sulla mesh anatomica
│
├── experiments/                          # Pipeline sperimentale end-to-end per la riproduzione dell'articolo
│   ├── 01_run_forward_verification.py    # Verifica MMS, bilanci di energia/massa e sensitività frazionaria
│   ├── 02_run_anatomical_forward.py      # Studio multifisico completo su geometria clinica BraTS
│   ├── 02_run_identifiability_study.py   # Studio SVD, correlazione locale, OAT e Profile Likelihood
│   ├── 03_run_twin_robustness.py         # Twin experiments multi-rumore (0%, 1%, 5%, 10%) anti-inverse-crime
│   └── 04_run_sota_benchmark.py          # Benchmark comparativo degli ottimizzatori a budget FEM fissato
│
├── tests/                                # Suite di verifica automatica (pytest)
│   ├── synthetic_brats_fixture.py        # Generatore NIfTI sintetico conforme a BraTS per test autonomi
│   ├── test_01_caputo_l1.py              # Verifica analitica dei pesi L1 su t^m e limite gamma -> 1
│   ├── test_02_initial_compatibility.py  # Verifica dell'equilibrio meccanico iniziale a t = 0
│   ├── test_03_energy_and_mass.py        # Verifica identità energetica discreta e conservazione M(t)=M(0)
│   ├── test_04_mms_coupled.py            # Verifica degli ordini di convergenza spaziali e temporali MMS
│   ├── test_05_identifiability_zero_q.py # Verifica Proposizione di non-identificabilità per q = 0
│   ├── test_06_inverse_consistency.py    # Verifica contatore valutazioni FEM, bounds e determinismo seed
│   ├── test_06_brats_anatomical_pipeline.py # Verifica estrazione NIfTI BraTS e meshing quadrilaterale Gmsh
│   ├── test_07_anatomical_mesh_integration.py # Verifica importazione .msh in deal.II e partizione dell'unità
│   ├── test_08_anatomical_forward_experiment.py # Verifica simulazione diretta e diagnostica su mesh anatomica
│   └── test_09_identifiability_study_revision.py # Verifica SVD multi-regime, correlazione, OAT e profili 1D
│
├── paper/                                # Sorgenti LaTeX e figure vettoriali del manoscritto scientifico
│   ├── paper.tex                         # Documento principale LaTeX
│   ├── paper.pdf                         # Manoscritto compilato
│   ├── references.bib                    # Bibliografia BibTeX
│   ├── compiler/                         # Utilità e configurazione di compilazione LaTeX
│   ├── sections/                         # Sezioni modulari del manoscritto
│   │   ├── section_1_introduction.tex
│   │   ├── section_2_forward.tex
│   │   ├── section_3_inverse.tex
│   │   ├── section_4_numerical.tex
│   │   ├── section_5_results.tex
│   │   └── section_6_conclusions.tex
│   └── images/                           # Figure PDF generate automaticamente dagli script in experiments/
│       ├── 01_forward_verification/
│       ├── 02_anatomical_forward_clinical/
│       └── 02_identifiability/
│
├── data/                                 # Dati NIfTI BraTS e mesh Gmsh (esclusi da git tramite .gitignore)
└── artifacts/                            # Certificati JSON, CSV e log delle esecuzioni sperimentali
```

---

## 5. Acquisizione dei Dati MRI (Dataset BraTS), Pipeline Anatomica e `.gitignore`

Per ragioni di dimensione dei file binari e di conformità alle licenze d'uso dei dataset neuroradiologici pubblici, i volumi di risonanza magnetica in formato NIfTI (`.nii`, `.nii.gz`) e i file intermedi in `data/` sono esclusi dal controllo di versione tramite il file `.gitignore`.

### 5.1. Nota metodologica sull'uso delle immagini anatomiche

Come precisato nella Sezione 1 e nella Sezione 5 del manoscritto (`paper/sections/`), una geometria estratta da un'immagine di risonanza magnetica definisce esclusivamente un **dominio computazionale bidimensionale realistico** $\Omega_h$ (tramite conversione affine da coordinate voxel a millimetri fisici e successiva adimensionalizzazione rispetto alla scala caratteristica $L_0$). Tale procedura **non** equivale a una serie temporale clinica longitudinale dei campi $(\mathbf{d}, p, c)$ né costituisce una validazione clinica o fisiologica *in vivo*: tutti gli esperimenti di identificazione inversa condotti in questo lavoro sono **studi metodologici su dati sintetici (*twin experiments*)**.

### 5.2. Fonti ufficiali per il download del dataset BraTS

I volumi multimodali (`t1`, `t1ce`, `t2`, `flair` e maschera di segmentazione `seg`) utilizzati per l'estrazione della sezione assiale parenchimale provengono dal **Brain Tumor Segmentation (BraTS) Challenge Dataset**:

1. **Portale Ufficiale Synapse (RSNA-ASNR-MICCAI BraTS Challenge)**:
   * URL: <https://www.synapse.org/brats>
2. **Center for Biomedical Image Computing and Analytics (CBICA — University of Pennsylvania)**:
   * URL: <https://www.med.upenn.edu/cbica/brats/>
3. **The Cancer Imaging Archive (TCIA) — BraTS Collection**:
   * URL: <https://www.cancerimagingarchive.net/>

### 5.3. Posizionamento dei file ed esecuzione della pipeline anatomica

1. Creare la struttura locale sotto `data/` (ignorata da Git) e copiare il soggetto BraTS configurato in `configs/brats_anatomical_forward.yaml`:

```bash
mkdir -p data/01_raw data/02_interim data/03_meshes
```

2. Eseguire la pipeline di estrazione anatomica e generazione delle mesh quadrilaterali conformi (ai tre livelli di risoluzione `coarse`, `medium`, `fine`):

```bash
source ./activate_reteco.sh
python scripts/run_brats_anatomical_pipeline.py --config configs/brats_anatomical_forward.yaml
python scripts/run_anatomical_forward_simulation.py --config configs/brats_anatomical_forward.yaml
```

3. **Modalità autonoma senza dataset esterno (`synthetic_brats_fixture.py`)**: per garantire che l'intera suite di test (`pytest`) sia eseguibile immediatamente su qualunque macchina o server di Continuous Integration anche in assenza dei volumi clinici BraTS in `data/`, il modulo `tests/synthetic_brats_fixture.py` genera al volo volumi NIfTI sintetici dotati di matrice affine realistica e maschere multi-classe conformi allo standard BraTS.

---

## 6. Requisiti di Sistema e Installazione Deterministica

L'infrastruttura supporta nativamente **macOS** (Apple Silicon `arm64` e Intel `x86_64`) e **Linux** (`x86_64`).

### 6.1. Prerequisiti software

* **Micromamba / Mamba / Conda**: per la gestione dell'ambiente isolato `./.conda_env` definito in `environment.yml`.
* **Libreria `deal.II` ($\ge 9.5.0$, testato e certificato con `9.8.0`)**: rilevata automaticamente da `prepare.sh` (sia da installazioni compilate tramite `candi` in `~/dealii-candi/deal.II-v9.8.0` o `/Applications/deal.II.app`, sia tramite `DEAL_II_DIR`).
* **Compilatore C++20**: Apple `clang++` (Xcode Command Line Tools su macOS) o `g++ >= 12` su Linux.

### 6.2. Procedura di installazione e compilazione automatica

Dalla directory radice del repository, eseguire:

```bash
# 1. Rendere eseguibile lo script di bootstrap
chmod +x prepare.sh

# 2. Configurare l'ambiente, verificare deal.II e compilare il modulo C++20 `reteco._core`
./prepare.sh --build

# 3. Attivare l'ambiente deterministico nella sessione terminale corrente
source ./activate_reteco.sh
```

**Operazioni svolte automaticamente da `./prepare.sh --build`:**

1. Crea o aggiorna l'ambiente isolato in `./.conda_env` installando tutte le dipendenze scientifiche Python (`numpy`, `scipy`, `sympy`, `pydantic`, `pybind11`, `gmsh`, `nibabel`, `scikit-image`, `matplotlib`, `pytest`) e gli strumenti di build (`cmake`, `ninja`).
2. Genera lo script `activate_reteco.sh` che imposta i percorsi di `deal.II` e forza il determinismo numerico single-thread (`PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`).
3. Compila in modalità `Release` la libreria statica `libreteco_cpp_core.a` e l'estensione binaria `python/reteco/_core.cpython-311-*.so`, inizializzando in modo sicuro il runtime MPI/Kokkos/TBB di `deal.II` tramite la guardia statica `DealIIRuntimeGuard`.
4. Registra il pacchetto Python `reteco` in modalità *editable* e salva il certificato di provenienza hardware/software in `artifacts/environment_provenance.json` e il lockfile dei pacchetti in `artifacts/conda_lock.json`.

---

## 7. Suite di Verifica Automatica (`pytest` — 10 Moduli di Test)

Prima di eseguire gli script sperimentali, la correttezza matematica, algebrica e geometrica dell'intera pipeline viene verificata tramite i 10 moduli della suite `tests/`.

Per eseguire tutti i test automatici:

```bash
source ./activate_reteco.sh
pytest tests/ -v -s
```

*(In alternativa: `make test`)*.

### Dettaglio dei moduli di test e criteri quantitativi di superamento

| File di Test | Componente / Proprietà Matematica Verificata | Criterio Quantitativo di Superamento (Asserzioni) |
| :--- | :--- | :--- |
| `tests/test_01_caputo_l1.py` | Formula di quadratura $L1$ per ${}^C D_t^\gamma u$ su griglie uniformi e graduate $t_n = T(n/N)^r$, positività dei pesi e limite classico $\gamma \uparrow 1$. | Errore su $u(t)=t$ pari a zero macchina ($< 10^{-13}$); ordine empirico di convergenza su $u(t)=t^2$ conforme a $2-\gamma$ ($\ge 1.90$); convergenza a $\partial_t u$ per $\gamma = 0.99999$. |
| `tests/test_02_initial_compatibility.py` | Condizione di compatibilità meccanica iniziale $a_E(\mathbf{d}_0, \mathbf{v}) - b(p_0, \nabla\cdot\mathbf{v}) = \langle F(0), \mathbf{v}\rangle$ e validazione rigorosa di `FixedPhysicsConfig`. | Norma $L^2$ discreta del residuo sulle componenti libere di $V$ inferiore a $10^{-11}$; rifiuto immediato (`ValueError`) di parametri non finiti, fuori dominio o con $T_0 \neq \tau_c$. |
| `tests/test_03_energy_and_mass.py` | Positività della dissipazione viscoelastica $\mathcal{D}_\beta(t) \ge 0$, identità energetica discreta passo-passo e conservazione della massa $M(t)=M(0)$. | Identità energetica discreta $\vert(R_n - R_{n-1}) + D_{\mathrm{num},n}\vert < 10^{-11}$ a ogni passo; decadimento $\mathcal{E}(t_n) + \mathcal{D}_\beta(t_n) \le \mathcal{E}(0)$ nel caso omogeneo; deriva relativa di massa $\vert M(t_n)-M(0)\vert / M(0) < 10^{-12}$. |
| `tests/test_04_mms_coupled.py` | Metodo delle Soluzioni Manufatte (MMS) sul sistema accoppiato completo $(\mathbf{d}^*, p^*, c^*)$ in $\Omega = (0,1)^2$. | Verifica esplicita degli ordini empirici di convergenza spaziale ($L^2$ e $H^1$ per $\mathbf{d}_h, p_h, c_h$) e temporale al raffinamento di $h$ e $\Delta t$. |
| `tests/test_05_identifiability_zero_q.py` | Proposizione di non-identificabilità di $(E, \beta)$ da osservazioni della sola concentrazione $c$ quando $\mathbf{q}_{\boldsymbol{\theta}} \equiv \mathbf{0}$. | Per $\mathbf{q} \equiv \mathbf{0}$: colonne del Jacobiano $\partial\mathcal{G}/\partial\widetilde{\beta}$ e $\partial\mathcal{G}/\partial\widetilde{E}$ nulle ($< 10^{-12}$) e rango numerico $\le 2$. Con forzante attiva ($\mathbf{q} \neq \mathbf{0}$): colonne non nulle e rango numerico pari a $4$. |
| `tests/test_06_inverse_consistency.py` | Invertibilità di $\widetilde{\boldsymbol{\theta}}=\psi(\boldsymbol{\theta})$, contatore immutabile `fem_evaluation_count`, arresto a budget esaurito e determinismo col seed. | Errore di inversione $\Vert\psi^{-1}(\psi(\boldsymbol{\theta})) - \boldsymbol{\theta}\Vert < 10^{-14}$; incremento esatto del contatore FEM; coincidenza bit-a-bit di due esecuzioni CPSO indipendenti a parità di `seed`. |
| `tests/test_06_brats_anatomical_pipeline.py` | Estrazione del contorno parenchimale da NIfTI BraTS, trasformazione affine in coordinate adimensionali e generazione di mesh quadrilaterali Gmsh. | Verifica di assenza di elementi invertiti (Jacobiano geometrico positivo su tutte le celle quadrilaterali) e corretta assegnazione dei `Physical Tags` per $\Gamma_D$, $\Gamma_N$ e dominio $\Omega_h$. |
| `tests/test_07_anatomical_mesh_integration.py` | Importazione di mesh `.msh` in `deal.II` (`InvariantFEMMatrices2D`) e verifica algebrica degli spazi Taylor–Hood $[Q_2]^2/Q_1$ e $Q_1$. | Superamento di `verify_dof_and_mesh_consistency`: le costanti discrete integrano esattamente l'area geometrica del dominio anatomico ($\mathbf{1}^{\mathsf{T}} \mathbf{M}_p \mathbf{1} = \mathbf{1}^{\mathsf{T}} \mathbf{M}_c \mathbf{1} = \vert\Omega_h\vert$ a $< 10^{-11}$). |
| `tests/test_08_anatomical_forward_experiment.py` | Esecuzione del solutore accoppiato completo su mesh anatomica BraTS con monitoraggio energetico e di massa. | Assenza di valori non finiti, compatibilità iniziale soddisfatta a $< 10^{-11}$, identità energetica discreta passo-passo verificata e conservazione esatta della massa del tracciante in assenza di sorgenti. |
| `tests/test_09_identifiability_study_revision.py` | Coerenza matematica dello studio di identificabilità esteso (SVD multi-osservabile, matrice di correlazione locale, OAT e *Profile Likelihood*). | Simmetria e diagonale unitaria della matrice di correlazione con elementi in $[-1, 1]$; minimo esatto in $\widetilde{\boldsymbol{\theta}}^\dagger$ delle sezioni OAT; disuguaglianza variazionale $\widehat{J}_j(\widetilde{\theta}_j) \le s_j(\widetilde{\theta}_j)$ verificata su tutti i nodi del profilo. |

---

## 8. Pipeline di Esecuzione degli Esperimenti e Figure del Manoscritto

Ogni script in `experiments/` genera sia i file dati certificati (`.json`, `.csv`, `.tex`) dentro `artifacts/`, sia le figure vettoriali definitive in `paper/images/` pronte per la compilazione di `paper/paper.tex`.

### 8.1. Esperimento 1 — Verifica Numerica del Problema Diretto e Sensitività Frazionaria

Esegue lo studio di convergenza spaziale e temporale MMS per $(\mathbf{d}^*, p^*, c^*)$, verifica l'identità energetica discreta e la conservazione della massa $M(t)$, analizza la sensibilità fisica della risposta agli ordini frazionari $(\alpha, \beta)$ e illustra l'accoppiamento multifisico preliminare.

* **Comando di esecuzione**:

```bash
source ./activate_reteco.sh
python experiments/01_run_forward_verification.py
```

* **Dati certificati generati**: `artifacts/01_forward_verification/forward_verification_summary.json`.
* **Figure generate in `paper/images/01_forward_verification/`**:
  * `fig_01_mms_convergence_complete.pdf` ( e `fig_mms_spatial_convergence.pdf`): curve log-log degli errori $L^2(\Omega)$ e $H^1(\Omega)$ rispetto alle pendenze teoriche per spostamento, pressione e concentrazione.
  * `fig_02_energy_and_mass_conservation.pdf` (e `fig_energy_mass_diagnostics.pdf`): evoluzione dell'energia meccanica/idraulica $\mathcal{E}(t_n)$, dissipazione viscoelastica di Caputo $\mathcal{D}_\beta(t_n)$, dissipazione di Darcy, residuo dell'identità energetica discreta e conservazione della massa $M(t_n)/M(0)$.
  * `fig_03_fractional_orders_physics_sensitivity.pdf`: effetto della memoria viscoelastica $\beta$ sul rilassamento di pressione/spostamento e dell'ordine sub-diffusivo $\alpha$ sulle code di ritenzione del tracciante.
  * `fig_04_brats_multiphysics_fields.pdf`: panoramica dei campi accoppiati $(\Vert\mathbf{d}_h\Vert, p_h, \Vert\mathbf{q}_h\Vert, c_h)$.

### 8.2. Esperimento 2A — Simulazione Multifisica Diretta su Geometria Anatomica Clinica (BraTS)

Esegue la pipeline completa su una sezione assiale reale del dataset BraTS (configurata in `configs/brats_anatomical_forward.yaml`), documentando l'estrazione multimodale, la qualità della mesh quadrilaterale su tre livelli di raffinamento (`coarse`, `medium`, `fine`), l'evoluzione temporale dei campi accoppiati, la sensibilità alla risoluzione spaziale e agli ordini frazionari, e la diagnostica energetica/massica sul dominio anatomico.

* **Comando di esecuzione**:

```bash
source ./activate_reteco.sh
python experiments/02_run_anatomical_forward.py --config configs/brats_anatomical_forward.yaml
```

* **Dati certificati generati**: `artifacts/02_anatomical_forward_clinical/` (metriche di qualità mesh, norme $L^2$ alle tre risoluzioni, serie temporali nei punti sonda e bilanci).
* **Figure generate in `paper/images/02_anatomical_forward_clinical/`**:
  * `fig01_multimodal_anatomy_and_annotations.pdf`: sequenze MRI multimodali (`T1`, `T1ce`, `T2`, `FLAIR`), annotazioni radiologiche e contorno parenchimale estratto.
  * `fig02_anatomical_mask_and_quad_mesh_quality.pdf`: partizione anatomica dei sottodomini, condizioni al contorno ($\Gamma_D, \Gamma_N$), mesh quadrilaterale conforme e distribuzione della qualità geometrica degli elementi.
  * `fig03_multiphysics_fields_on_quad_mesh.pdf`: mappe spaziali ad alta risoluzione di $\Vert\mathbf{d}_h\Vert$, $p_h$, campo vettoriale di Darcy $\mathbf{q}_h$ e concentrazione $c_h$ sulla mesh anatomica.
  * `fig04_temporal_evolution_comparable_scales.pdf`: istantanee temporali su scale cromatiche uniformi per il confronto diretto dell'evoluzione dinamica.
  * `fig05_mesh_resolution_sensitivity.pdf`: confronto quantitativo tra le risoluzioni `coarse`, `medium` e `fine` lungo transetti spaziali e sonde temporali.
  * `fig06_fractional_parameters_sensitivity.pdf`: risposta del dominio anatomico alla variazione degli ordini frazionari $(\alpha, \beta)$.
  * `fig07_energy_and_mass_diagnostics.pdf`: verifica dell'identità energetica discreta e del bilancio di massa sulla geometria anatomica.

### 8.3. Esperimento 2B — Studio di Identificabilità Locale, Correlazione e *Profile Likelihood*

Analizza l'identificabilità del vettore parametrico $\boldsymbol{\theta}=(\alpha,\beta,D,E)^{\mathsf{T}}$ confrontando il caso patologico a flusso nullo ($\mathbf{q}\equiv\mathbf{0}$) con tre regimi osservativi a flusso attivo: sola concentrazione ($c$), multimodale idro-chimico ($c, p$) e multimodale completo ($c, p, \mathbf{d}$).

* **Comando di esecuzione**:

```bash
source ./activate_reteco.sh
python experiments/02_run_identifiability_study.py
```

* **Dati certificati generati**: `artifacts/02_identifiability/identifiability_study_summary.json`.
* **Figure generate in `paper/images/02_identifiability/`**:
  * `figA_sensitivity_and_svd.pdf`: norme delle colonne di sensibilità normalizzate $\Vert\partial\mathcal{G}/\partial\widetilde{\theta}_j\Vert_2$ e spettro dei valori singolari normalizzati $\sigma_k/\sigma_1$ nei quattro regimi osservativi.
  * `figB_local_correlation.pdf`: matrici di correlazione locale derivanti dall'informazione di Fisher e composizione assoluta dei vettori singolari destri $\vert V_{jk}\vert$.
  * `figC_one_at_a_time_cost_slices.pdf`: profili *One-At-a-Time* (OAT) del funzionale obiettivo $J_h(\widetilde{\theta}_j, \widetilde{\boldsymbol{\theta}}_{-j}^\dagger)$ attorno a $\widetilde{\boldsymbol{\theta}}^\dagger$.
  * `figD_optimized_objective_profiles.pdf`: confronto diretto tra sezioni OAT bloccate e *Profile Likelihood* ri-ottimizzata $\widehat{J}_j(\widetilde{\theta}_j)$ al variare dei campi osservati.

### 8.4. Esperimento 3 — *Twin Experiments* Anti-Inverse-Crime e Robustezza al Rumore

Genera i dati sintetici $\mathbf{y} = \mathcal{G}_{h_{\mathrm{tar}}}(\boldsymbol{\theta}^\dagger) + \boldsymbol{\epsilon}$ sulla mesh fine ($N_{\mathrm{tar}}=32, M_{\mathrm{tar}}=160$), quantifica la discrepanza di discretizzazione rispetto alla mesh di inversione ($N_{\mathrm{inv}}=16, M_{\mathrm{inv}}=80$) ed esegue l'inversione multi-replica ai livelli di rumore $\eta \in \{0\%, 1\%, 5\%, 10\%\}$.

* **Comando di esecuzione**:

```bash
source ./activate_reteco.sh
python experiments/03_run_twin_robustness.py
```

* **Output generati in `artifacts/03_twin_robustness/`**:
  * `twin_robustness_results.json`: discrepanza di discretizzazione anti-inverse-crime, statistiche (media, deviazione standard, mediana, IQR) degli errori relativi percentuali su $(\alpha, \beta, D, E)$ e tracce complete delle repliche.
  * `fig_twin_robustness_boxplot.pdf`: distribuzione degli errori di ricostruzione parametrica in funzione del livello di rumore $\eta$.

### 8.5. Esperimento 4 — Benchmark Comparativo SOTA a Budget FEM Fissato

Confronta in condizioni di parità (`max_fem_evaluations = 240`, stessa istanza del funzionale $J_h$, stessi limiti $\Theta$ e repliche multi-seed) l'algoritmo `CPSO` con `L-BFGS-B`, `Powell`, `Nelder-Mead` e `Differential-Evolution`.

* **Comando di esecuzione**:

```bash
source ./activate_reteco.sh
python experiments/04_run_sota_benchmark.py
```

* **Output generati in `artifacts/04_sota_benchmark/`**:
  * `sota_benchmark_results.json`: registro certificato di tutte le repliche, numero esatto di valutazioni FEM consumate, tempi di calcolo e statistiche comparative.
  * `table_sota_benchmark.tex`: tabella LaTeX generata automaticamente dai risultati numerici certificati.
  * `fig_sota_convergence.pdf`: curve di convergenza del miglior valore di $J_h$ in funzione del budget di chiamate al solutore FEM.

---

## 9. Struttura e Compilazione del Manoscritto (`paper/`)

Il manoscritto dell'articolo è interamente contenuto nella directory `paper/` ed è strutturato in modo modulare:

* `paper/paper.tex`: file master LaTeX che include le 6 sezioni tematiche e la bibliografia `paper/references.bib`.
* `paper/sections/section_1_introduction.tex`: inquadramento biomeccanico poro-viscoelastico frazionario, ipotesi di accoppiamento unidirezionale e obiettivi metodologici.
* `paper/sections/section_2_forward.tex`: equazioni dimensionali, adimensionalizzazione ($T_0=\tau_c$), formulazione debole, identità energetica (Proposizione 1), conservazione della massa e stabilità di Grönwall–Mittag-Leffler.
* `paper/sections/section_3_inverse.tex`: operatore osservativo $\mathcal{H}$, funzionale regolarizzato $J_h(\boldsymbol{\theta})$, trasformazione $\widetilde{\boldsymbol{\theta}}=\psi(\boldsymbol{\theta})$, identificabilità locale, Proposizione di non-identificabilità a flusso nullo (Proposizione 2) e protocollo di inversione.
* `paper/sections/section_4_numerical.tex`: schema temporale $L1$ su griglie graduate, elementi finiti misti Taylor–Hood $[Q_2]^2/Q_1$ + $Q_1$, formulazione analitica del test MMS accoppiato e pipeline di generazione delle mesh quadrilaterali anatomiche BraTS.
* `paper/sections/section_5_results.tex`: presentazione e discussione critica dei risultati numerici di verifica diretta, simulazione anatomica BraTS, studio di identificabilità SVD/OAT/Profile-Likelihood, *twin experiments* e benchmark SOTA.
* `paper/sections/section_6_conclusions.tex`: conclusioni metodologiche, limiti di identificabilità osservati e sviluppi futuri (estensione 3D su volumi interi e validazione multimodale).

---

## 10. Tracciabilità, Criteri Anti-Inverse-Crime e Limiti Metodologici

In conformità con il contratto matematico (`contratto_matematico.md`) e con il rigore espositivo del manoscritto, l'utilizzo del codice e l'interpretazione dei risultati rispettano quattro vincoli invalicabili:

1. **Separazione tra stime per soluzioni regolari ed esistenza debole**: le proprietà energetiche e di unicità dimostrate nella Sezione 2 e verificate discretamente in `energy_mass_monitor.cpp` valgono nella classe di regolarità specificata; esse non vengono mai presentate come dimostrazione di esistenza debole in spazi minimi per il sistema continuo accoppiato completo.
2. **Prevenzione dell'*inverse crime***: in `configs/twin_experiment_synthetic.yaml`, la discretizzazione spaziale e temporale usata per generare l'osservazione sintetica $\mathbf{y} = \mathcal{G}_{h_{\mathrm{tar}}}(\boldsymbol{\theta}^\dagger) + \boldsymbol{\epsilon}$ è strettamente più fine rispetto a quella impiegata dall'ottimizzatore in $\mathcal{G}_{h_{\mathrm{inv}}}(\boldsymbol{\theta})$. La discrepanza numerica $\Vert\mathcal{G}_{h_{\mathrm{tar}}}(\boldsymbol{\theta}^\dagger) - \mathcal{G}_{h_{\mathrm{inv}}}(\boldsymbol{\theta}^\dagger)\Vert_2$ viene sempre misurata e rendicontata esplicitamente.
3. **Parità assoluta nel benchmark degli ottimizzatori**: tutti gli algoritmi in `python/reteco/inverse/sota_runners.py` interrogano la medesima istanza di `ObjectiveFunctional`, che contabilizza ogni singola esecuzione di `solve_trajectory` e arresta l'ottimizzatore tramite `BudgetExhaustedException` al raggiungimento esatto di `max_fem_evaluations`.
4. **Distinzione tra identificazione sintetica su geometria anatomica e validazione clinica**: le simulazioni sulla geometria BraTS dimostrano l'applicabilità dello schema numerico su domini anatomici realistici con eterogeneità geometrica, ma non sostituiscono misure cliniche longitudinali indipendenti di spostamento, pressione interstiziale e concentrazione.

---

## 11. Citazione e Riferimenti Bibliografici

Se utilizzi questo codice, la pipeline anatomica o la formulazione del modello **RETECO v1** nel tuo lavoro di ricerca, sei pregato di citare il manoscritto di riferimento:

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