#!/usr/bin/env bash
set -u

if [ -f "./activate_reteco.sh" ]; then
    # shellcheck disable=SC1091
    source ./activate_reteco.sh
fi

OUT_DIR="artifacts/09_beta_profile"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/phase39_execution.log"
: > "${SUMMARY_LOG}"

run_and_log() {
    local step_name="$1"
    local cmd="$2"
    local log_file="${LOG_DIR}/${step_name}.log"
    echo "============================================================" | tee -a "${SUMMARY_LOG}"
    echo "[CMD] ${cmd}" | tee -a "${SUMMARY_LOG}"
    echo "[START UTC] $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${SUMMARY_LOG}"
    eval "${cmd}" > "${log_file}" 2>&1
    local ec=$?
    cat "${log_file}" | tee -a "${SUMMARY_LOG}"
    echo "[EXIT_CODE] ${ec}" | tee -a "${SUMMARY_LOG}"
    echo "[END UTC] $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${SUMMARY_LOG}"
    if [ ${ec} -ne 0 ]; then
        echo "[ABORT] Fallimento nello step ${step_name} (exit code ${ec}). Interruzione immediata." | tee -a "${SUMMARY_LOG}"
        exit ${ec}
    fi
}

# 1. Estensione non distruttiva di python/reteco/identifiability/profile_likelihood.py
run_and_log "01_update_profile_likelihood" \
    "python scripts/update_profile_likelihood_phase39.py"

# 2. Esecuzione dei test esistenti sull'identificabilità/forzanti e del nuovo test Fase 3.9
run_and_log "02_test_phase39_beta_profile" \
    "pytest -v tests/test_09_*.py tests/test_14_phase39_beta_profile.py"

# 3. Esecuzione dell'analisi dei due profili ottimizzati Phi_data(beta) e Phi_lambda(beta)
run_and_log "03_run_beta_profile_analysis" \
    "python experiments/09_beta_profile_analysis.py"