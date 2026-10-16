#!/usr/bin/env bash
# ==============================================================================
# reteco-v1/prepare.sh
# Prepara l'ambiente di lavoro:
#   1) ambiente conda isolato in ./.conda_env (da environment.yml)
#   2) collegamento a deal.II (installato con candi, NON da conda) e verifica
#
# Uso:
#   ./prepare.sh            solo ambiente conda + configurazione deal.II
#   ./prepare.sh --build    in più compila il modulo C++ e installa il pacchetto
#
# deal.II non è disponibile su conda-forge per macOS Apple Silicon: va installato
# prima con candi (https://github.com/dealii/candi). Se non si trova in
# ~/dealii-candi/deal.II-*, indica il percorso così:
#   DEAL_II_DIR=/percorso/a/deal.II-vX.Y.Z ./prepare.sh
# ==============================================================================

set -euo pipefail

usage() {
    cat <<'USAGE'
Uso: ./prepare.sh [--build]

  (nessuna opzione)  crea/aggiorna l'ambiente conda e verifica deal.II
  --build            in più compila reteco._core e installa il pacchetto Python

Variabili opzionali:
  DEAL_II_DIR        percorso dell'installazione deal.II (default: l'ultima in ~/dealii-candi)
USAGE
}

DO_BUILD=0
for arg in "$@"; do
    case "${arg}" in
        --build) DO_BUILD=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Argomento sconosciuto: ${arg}"; usage; exit 1 ;;
    esac
done

DEALII_MIN_VERSION="9.5.1"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${PROJECT_ROOT}/.conda_env"
BUILD_DIR="${PROJECT_ROOT}/build"
ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts"
LOCAL_BIN="${PROJECT_ROOT}/.local/bin"

echo "======================================================================"
echo " RETECO v1 — Setup ambiente conda + deal.II (macOS)"
echo "======================================================================"

# 1. Controllo Sistema Operativo e Xcode Command Line Tools
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
echo "[1/6] Rilevato sistema: ${OS_NAME} (${ARCH_NAME})"

if [[ "${OS_NAME}" == "Darwin" ]]; then
    if ! xcode-select -p &>/dev/null; then
        echo "ERRORE: Xcode Command Line Tools non trovati su macOS."
        echo "Esegui: xcode-select --install e poi rilancia ./prepare.sh"
        exit 1
    fi
    SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
    export SDKROOT
    echo "      macOS SDK rilevato: ${SDKROOT}"
fi

# 2. Individuazione di deal.II (installato con candi). Controllo subito, così
#    se manca non si perde tempo a creare l'ambiente conda.
DEALII_PREFIX="${DEAL_II_DIR:-}"
if [[ -z "${DEALII_PREFIX}" ]]; then
    DEALII_PREFIX="$(ls -d "${HOME}"/dealii-candi/deal.II-* 2>/dev/null | sort -V | tail -n1 || true)"
fi
if [[ -z "${DEALII_PREFIX}" || ! -d "${DEALII_PREFIX}/lib/cmake/deal.II" ]]; then
    echo "ERRORE: deal.II non trovato."
    echo "  Cercato in: ${DEAL_II_DIR:-${HOME}/dealii-candi/deal.II-*}"
    echo "  Installalo con candi (https://github.com/dealii/candi) oppure esporta DEAL_II_DIR."
    exit 1
fi
echo "[2/6] deal.II trovato: ${DEALII_PREFIX}"

# 3. Individuazione o installazione locale di Micromamba
mkdir -p "${LOCAL_BIN}" "${ARTIFACTS_DIR}"

MAMBA_CMD=""
if command -v micromamba &>/dev/null; then
    MAMBA_CMD="micromamba"
elif [[ -x "${LOCAL_BIN}/micromamba" ]]; then
    MAMBA_CMD="${LOCAL_BIN}/micromamba"
elif command -v mamba &>/dev/null; then
    MAMBA_CMD="mamba"
elif command -v conda &>/dev/null; then
    MAMBA_CMD="conda"
else
    echo "[3/6] Nessun gestore Conda/Mamba trovato. Scarico micromamba locale..."
    if [[ "${ARCH_NAME}" == "arm64" ]]; then
        MM_ARCH="osx-arm64"
    else
        MM_ARCH="osx-64"
    fi
    curl -Ls "https://micro.mamba.pm/api/micromamba/${MM_ARCH}/latest" | tar -xvj -C "${LOCAL_BIN}" --strip-components=1 bin/micromamba
    MAMBA_CMD="${LOCAL_BIN}/micromamba"
fi
echo "[3/6] Gestore pacchetti selezionato: ${MAMBA_CMD}"

# 4. Creazione o aggiornamento dell'ambiente locale ./.conda_env
if [[ ! -d "${ENV_PREFIX}/conda-meta" ]]; then
    echo "[4/6] Creazione dell'ambiente isolato in ${ENV_PREFIX}..."
    "${MAMBA_CMD}" create -y -p "${ENV_PREFIX}" -f "${PROJECT_ROOT}/environment.yml"
else
    echo "[4/6] Ambiente esistente trovato in ${ENV_PREFIX}. Aggiornamento dipendenze..."
    "${MAMBA_CMD}" env update -p "${ENV_PREFIX}" -f "${PROJECT_ROOT}/environment.yml"
fi

# 5. Generazione dello script di attivazione (activate_reteco.sh) e verifica deal.II.
#    activate_reteco.sh NON esiste prima di questo passaggio: lo crea questo script.
echo "[5/6] Generazione di activate_reteco.sh e verifica di deal.II..."
cat > "${PROJECT_ROOT}/activate_reteco.sh" <<EOF
# File generato da prepare.sh: non modificarlo a mano.
# Usalo con: source ./activate_reteco.sh
export PATH="${ENV_PREFIX}/bin:\${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
export DEAL_II_DIR="${DEALII_PREFIX}"
export CMAKE_PREFIX_PATH="${ENV_PREFIX}:${DEALII_PREFIX}"
# Variabili per la riproducibilità numerica bit-a-bit
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}/python:\${PYTHONPATH:-}"
EOF

# Attiviamo subito le variabili nel processo corrente
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/activate_reteco.sh"

# Configure di prova: verifica che CMake trovi deal.II >= versione minima.
# (solo configurazione, nessuna compilazione)
CHECK_DIR="$(mktemp -d)"
cat > "${CHECK_DIR}/CMakeLists.txt" <<CMAKEEOF
cmake_minimum_required(VERSION 3.13)
find_package(deal.II ${DEALII_MIN_VERSION} REQUIRED HINTS "${DEALII_PREFIX}")
deal_ii_initialize_cached_variables()
project(reteco_dealii_check CXX)
message(STATUS "deal.II \${deal.II_VERSION} trovato in \${deal.II_DIR}")
CMAKEEOF

if ! CHECK_OUT="$(cmake -S "${CHECK_DIR}" -B "${CHECK_DIR}/build" 2>&1)"; then
    echo "${CHECK_OUT}"
    echo "ERRORE: CMake non riesce a configurare deal.II >= ${DEALII_MIN_VERSION} da ${DEALII_PREFIX}"
    rm -rf "${CHECK_DIR}"
    exit 1
fi
rm -rf "${CHECK_DIR}"
echo "${CHECK_OUT}" | grep -E "deal\.II .* trovato" | sed 's/^-- /      /' || true

# 6. Provenance e lockfile per il Paper
echo "[6/6] Salvataggio metadati di riproducibilità in artifacts/environment_provenance.json..."
"${MAMBA_CMD}" list -p "${ENV_PREFIX}" --json > "${ARTIFACTS_DIR}/conda_lock.json"

PROJECT_ROOT="${PROJECT_ROOT}" DEALII_PREFIX="${DEALII_PREFIX}" "${ENV_PREFIX}/bin/python" - <<'PYCODE'
import datetime
import json
import os
import platform
import subprocess
from pathlib import Path

root = Path(os.environ["PROJECT_ROOT"])
dealii_prefix = os.environ["DEALII_PREFIX"]

git_commit = "UNTRACKED"
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    pass

provenance = {
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "os": platform.platform(),
    "architecture": platform.machine(),
    "python_version": platform.python_version(),
    "git_commit": git_commit,
    "dealii": {
        "prefix": dealii_prefix,
        "install_dir_name": Path(dealii_prefix).name,
        "installed_with": "candi",
    },
    "deterministic_env": {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    },
}
out_file = root / "artifacts" / "environment_provenance.json"
out_file.write_text(json.dumps(provenance, indent=2))
print(f"      Certificato di riproducibilità scritto in: {out_file}")
PYCODE

# Opzionale: compilazione del motore C++20 + installazione del pacchetto Python
if [[ "${DO_BUILD}" -eq 1 ]]; then
    if [[ ! -f "${PROJECT_ROOT}/CMakeLists.txt" ]]; then
        echo "ERRORE: --build richiesto ma manca ${PROJECT_ROOT}/CMakeLists.txt"
        exit 1
    fi

    # Stesso compilatore usato da candi per deal.II (Apple clang): niente compilatori conda.
    COMPILER_ARGS=()
    if [[ "${OS_NAME}" == "Darwin" ]]; then
        COMPILER_ARGS=(-DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++)
    fi

    echo "[build 1/3] Configurazione CMake..."
    mkdir -p "${BUILD_DIR}"
    cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DDEAL_II_DIR="${DEALII_PREFIX}" \
        -DPython3_EXECUTABLE="${ENV_PREFIX}/bin/python" \
        ${COMPILER_ARGS[@]+"${COMPILER_ARGS[@]}"}

    echo "[build 2/3] Compilazione del modulo C++ (reteco._core)..."
    cmake --build "${BUILD_DIR}" --parallel

    echo "[build 3/3] Registrazione del pacchetto Python 'reteco'..."
    "${ENV_PREFIX}/bin/python" -m pip install --no-deps --no-build-isolation -e "${PROJECT_ROOT}"
fi

echo "======================================================================"
echo " Setup completato con successo!"
echo " Per lavorare nel terminale esegui: source ./activate_reteco.sh"
if [[ "${DO_BUILD}" -eq 0 ]]; then
    echo " Quando vuoi compilare il progetto: ./prepare.sh --build"
fi
echo "======================================================================"
