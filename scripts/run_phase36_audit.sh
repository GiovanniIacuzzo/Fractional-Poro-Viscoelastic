#!/usr/bin/env bash
set -u

if [ -f "./activate_reteco.sh" ]; then
    # shellcheck disable=SC1091
    source ./activate_reteco.sh
fi

LOG_DIR="artifacts/06_cpso_budget_diagnostic/test_logs"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/test_execution_summary.log"
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

# 1. Test focalizzati su CPSO, trasformazioni coordinate, funzionale obiettivo e isolamento
run_and_log "01_focused_inverse_tests" \
    "pytest -v tests/test_06_inverse_consistency.py tests/test_07_twin_diagnostics.py tests/test_09_twin_forcing_consistency.py tests/test_10_objective_functional_modes.py tests/test_11_phase36_audit_and_isolation.py"

# 2. Suite completa dei test
run_and_log "02_full_test_suite" \
    "pytest -v"

# 3. Esperimento controllato a 600 valutazioni FEM (solo caso senza rumore)
run_and_log "03_cpso_600_budget_experiment" \
    "python experiments/06_cpso_budget_diagnostic.py"