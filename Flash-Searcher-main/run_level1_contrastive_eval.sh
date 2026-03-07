#!/bin/bash
# ============================================================
# Level-1 Full Evaluation — Contrastive vs Agent-KB Baseline
# ============================================================
# Runs the full GAIA Level-1 (53 tasks) with:
#   0. no_memory          — pure LLM baseline (no memory provider)
#   1. agent_kb           — Agent-KB baseline (existing MembankEvolve method)
#   2. contrastive_min005 — Contrastive + tip+insight + retriever min_score=0.05
#   3. contrastive_minrel010 — Contrastive + tip+insight + provider min_relevance=0.10
#
# This tests our optimized modular architecture against:
#   - No memory (pure LLM ability)
#   - Agent-KB (existing best practice from MembankEvolve repo)
#
# Usage:
#   chmod +x run_level1_contrastive_eval.sh
#   ./run_level1_contrastive_eval.sh
#
# Override:
#   TASK_RANGE="1-10" ./run_level1_contrastive_eval.sh   # quick test
#   CONFIGS="0,1"     ./run_level1_contrastive_eval.sh   # subset
#   CONFIGS="0,1,2,3" ./run_level1_contrastive_eval.sh   # all configs
# ============================================================

set -euo pipefail

# ---------- Configurable parameters ----------
TASK_RANGE="${TASK_RANGE:-1-53}"       # all level-1 tasks
MAX_STEPS="${MAX_STEPS:-40}"
CONFIGS="${CONFIGS:-0,1,2,3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="./experiments/level1_contrastive_${TIMESTAMP}"
INFILE="./data/gaia/validation/metadata.jsonl"

# ---------- Environment ----------
eval "$(conda shell.bash hook)"
conda activate memevolve

if [ -z "${OPENAI_API_KEY:-}" ]; then
    source .env 2>/dev/null || true
fi

# ---------- Verify dataset ----------
if [ ! -f "${INFILE}" ]; then
    echo "[ERROR] GAIA dataset not found: ${INFILE}"
    echo "        Run: python download_gaia.py"
    exit 1
fi

# ---------- Print header ----------
echo "============================================================"
echo "  Level-1 Full Evaluation — Contrastive vs Agent-KB"
echo "============================================================"
echo "  Time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Task range:  [level1]${TASK_RANGE}"
echo "  Max steps:   ${MAX_STEPS}"
echo "  Configs:     ${CONFIGS}"
echo "  Output dir:  ${EVAL_DIR}"
echo "  Model:       ${DEFAULT_MODEL:-qwen3-max}"
echo "============================================================"
echo ""
echo "  Config map:"
echo "    0 = no_memory            (pure LLM, no memory)"
echo "    1 = agent_kb             (Agent-KB baseline from MembankEvolve)"
echo "    2 = contrastive_min005   (Contrastive + tip+insight, retriever min_score=0.05)"
echo "    3 = contrastive_minrel010 (Contrastive + tip+insight, provider min_relevance=0.10)"
echo ""

mkdir -p "${EVAL_DIR}"

# ---------- Generic experiment runner ----------
run_experiment() {
    local config_id="$1"
    local name="$2"
    local memory_provider="$3"
    shift 3
    # Remaining args are env var assignments

    local run_dir="${EVAL_DIR}/${name}"
    local storage_dir="${EVAL_DIR}/storage_${name}"
    local log_file="${run_dir}/run.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${config_id}] ${name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mkdir -p "${run_dir}"
    rm -rf "${storage_dir}"
    mkdir -p "${storage_dir}"

    local start_time=$(date +%s)

    if [ "${memory_provider}" = "none" ]; then
        echo "  Running without memory provider..."
        python run_flash_searcher_mm_gaia.py \
            --infile "${INFILE}" \
            --outfile "${run_dir}/results.jsonl" \
            --task_indices "[level1]${TASK_RANGE}" \
            --concurrency 1 \
            --max_steps "${MAX_STEPS}" \
            --direct_output_dir "${run_dir}" \
            2>&1 | tee "${log_file}"
    else
        echo "  Memory: ${memory_provider} | Storage: ${storage_dir}"
        env "$@" \
        python run_flash_searcher_mm_gaia.py \
            --infile "${INFILE}" \
            --outfile "${run_dir}/results.jsonl" \
            --task_indices "[level1]${TASK_RANGE}" \
            --memory_provider "${memory_provider}" \
            --enable_memory_evolution \
            --concurrency 1 \
            --max_steps "${MAX_STEPS}" \
            --direct_output_dir "${run_dir}" \
            2>&1 | tee "${log_file}"
    fi

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo "  [${name}] Completed in ${elapsed}s"

    # Count results
    local total=$(ls "${run_dir}"/*.json 2>/dev/null | grep -cv results || echo 0)
    local correct=$(grep -rl '"judgement".*"[Cc]orrect"' "${run_dir}"/*.json 2>/dev/null | wc -l || echo 0)

    echo "  [${name}] Tasks: ${total}, Correct: ${correct}"
    echo "${name},${total},${correct},${elapsed}" >> "${EVAL_DIR}/summary.csv"
}

# ---------- Initialize summary ----------
echo "config,tasks,correct,elapsed_seconds" > "${EVAL_DIR}/summary.csv"

# ---------- Run selected configurations ----------
IFS=',' read -ra CONFIG_ARRAY <<< "${CONFIGS}"

for cfg_id in "${CONFIG_ARRAY[@]}"; do
    cfg_id=$(echo "${cfg_id}" | tr -d ' ')
    case "${cfg_id}" in
        0)
            # Baseline: no memory at all
            run_experiment 0 "no_memory" "none"
            ;;
        1)
            # Agent-KB baseline: the existing method from MembankEvolve repo
            # Uses AGENT_KB_DATABASE_PATH env var for storage isolation
            run_experiment 1 "agent_kb" "agent_kb" \
                AGENT_KB_DATABASE_PATH="${EVAL_DIR}/storage_agent_kb/agent_kb_database.json"
            ;;
        2)
            # Contrastive + tip+insight + retriever-level min_score=0.05
            # Filters within each pool (success/failure) before weighting
            run_experiment 2 "contrastive_min005" "modular" \
                MODULAR_ENABLED_PROMPTS="tip,insight" \
                MODULAR_STORAGE_TYPE="json" \
                MODULAR_RETRIEVER_TYPE="contrastive" \
                MODULAR_RETRIEVER_CONFIG='{"success_weight":0.6,"failure_weight":0.4,"success_k":3,"failure_k":2,"min_score":0.05}' \
                MODULAR_MIN_RELEVANCE="0.0" \
                MODULAR_STORAGE_DIR="${EVAL_DIR}/storage_contrastive_min005"
            ;;
        3)
            # Contrastive + tip+insight + provider-level min_relevance=0.10
            # Filters after retriever scoring: discard memories with weighted score < 0.10
            run_experiment 3 "contrastive_minrel010" "modular" \
                MODULAR_ENABLED_PROMPTS="tip,insight" \
                MODULAR_STORAGE_TYPE="json" \
                MODULAR_RETRIEVER_TYPE="contrastive" \
                MODULAR_RETRIEVER_CONFIG='{"success_weight":0.6,"failure_weight":0.4,"success_k":3,"failure_k":2,"min_score":0.0}' \
                MODULAR_MIN_RELEVANCE="0.10" \
                MODULAR_STORAGE_DIR="${EVAL_DIR}/storage_contrastive_minrel010"
            ;;
        *)
            echo "[WARN] Unknown config id: ${cfg_id}, skipping"
            ;;
    esac
done

# ---------- Final comparison report ----------
echo ""
echo "============================================================"
echo "  EVALUATION COMPLETE"
echo "============================================================"
echo ""
echo "  Results directory: ${EVAL_DIR}"
echo ""

if [ -f "${EVAL_DIR}/summary.csv" ]; then
    echo "  ┌──────────────────────────────┬───────┬─────────┬──────────┬──────────┐"
    echo "  │ Configuration                │ Tasks │ Correct │ Accuracy │ Time (s) │"
    echo "  ├──────────────────────────────┼───────┼─────────┼──────────┼──────────┤"

    tail -n +2 "${EVAL_DIR}/summary.csv" | while IFS=',' read -r name tasks correct elapsed; do
        if [ "${tasks}" -gt 0 ] 2>/dev/null; then
            accuracy=$(echo "scale=1; ${correct} * 100 / ${tasks}" | bc 2>/dev/null || echo "N/A")
        else
            accuracy="N/A"
        fi
        printf "  │ %-28s │ %5s │ %7s │ %6s%% │ %8s │\n" \
            "${name}" "${tasks}" "${correct}" "${accuracy}" "${elapsed}"
    done

    echo "  └──────────────────────────────┴───────┴─────────┴──────────┴──────────┘"
fi

echo ""
echo "  Detailed results per config in: ${EVAL_DIR}/<config_name>/"
echo "  Summary CSV: ${EVAL_DIR}/summary.csv"
echo "============================================================"
