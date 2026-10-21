#!/usr/bin/env bash
set -u

if [ -f "./activate_reteco.sh" ]; then
    # shellcheck disable=SC1091
    source ./activate_reteco.sh
fi

OUT_DIR="artifacts/07_cpso_multiseed"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/phase37_execution.log"
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

# 1. Test unitario del protocollo multi-seed Fase 3.7
run_and_log "01_test_phase37_protocol" \
    "pytest -v tests/test_12_phase37_multiseed_protocol.py"

# 2. Esecuzione dell'esperimento multi-seed (riutilizzo seed 20261017 + nuove run 20261018 e 20261019)
run_and_log "02_run_cpso_multiseed_600" \
    "python experiments/07_cpso_multiseed_variability.py"