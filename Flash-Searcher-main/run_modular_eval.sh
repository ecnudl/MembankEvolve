#!/bin/bash
# ============================================================
# Modular Memory Evaluation — Ablation study on GAIA
# ============================================================
# Runs multiple memory configurations sequentially and compares
# accuracy to validate the decoupled Extraction×Storage×Retrieval
# architecture.
#
# Configurations:
#   0. no_memory            — baseline (no memory provider)
#   1. original_semantic    — tip only + JsonStorage + SemanticRetriever
#   2. tip_insight_semantic — tip+insight + JsonStorage + SemanticRetriever
#   3. tip_insight_hybrid   — tip+insight + JsonStorage + HybridRetriever
#   4. tip_insight_contrastive — tip+insight + JsonStorage + ContrastiveRetriever
#   5. tip_insight_graph    — tip+insight + GraphStore + GraphRetriever
#
# Usage:
#   chmod +x run_modular_eval.sh
#   ./run_modular_eval.sh
#
# Override defaults:
#   TASK_RANGE="1-20" ./run_modular_eval.sh          # quick test
#   CONFIGS="0,2,3"  ./run_modular_eval.sh           # subset of configs
#   MAX_STEPS=30     ./run_modular_eval.sh           # fewer steps
# ============================================================

set -euo pipefail

# ---------- Configurable parameters ----------
TASK_RANGE="${TASK_RANGE:-1-53}"       # level-1 tasks by default
MAX_STEPS="${MAX_STEPS:-40}"
CONFIGS="${CONFIGS:-0,1,2,3,4,5}"      # which configs to run (comma-sep)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="./experiments/modular_eval_${TIMESTAMP}"
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
echo "  Modular Memory Evaluation — Ablation Study"
echo "============================================================"
echo "  Time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Task range:  ${TASK_RANGE}"
echo "  Max steps:   ${MAX_STEPS}"
echo "  Configs:     ${CONFIGS}"
echo "  Output dir:  ${EVAL_DIR}"
echo "  Model:       ${DEFAULT_MODEL:-qwen3-max}"
echo "============================================================"
echo ""

mkdir -p "${EVAL_DIR}"

# ---------- Define experiment configurations ----------
# Format: run_experiment <config_id> <name> <memory_provider> <env_vars...>
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

    # Clean storage for fresh start
    rm -rf "${storage_dir}"
    mkdir -p "${storage_dir}"

    local start_time=$(date +%s)

    if [ "${memory_provider}" = "none" ]; then
        # No memory baseline
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
        # With memory provider
        echo "  Memory: ${memory_provider} | Storage: ${storage_dir}"
        env "$@" \
        MODULAR_STORAGE_DIR="${storage_dir}" \
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
    local total=$(ls "${run_dir}"/*.json 2>/dev/null | grep -v results | wc -l || echo 0)
    local correct=$(grep -rl '"judgement".*"[Cc]orrect"' "${run_dir}"/*.json 2>/dev/null | wc -l || echo 0)

    echo "  [${name}] Tasks: ${total}, Correct: ${correct}"

    # Save summary
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
            run_experiment 0 "no_memory" "none"
            ;;
        1)
            run_experiment 1 "original_semantic" "modular" \
                MODULAR_ENABLED_PROMPTS="tip" \
                MODULAR_STORAGE_TYPE="json" \
                MODULAR_RETRIEVER_TYPE="semantic"
            ;;
        2)
            run_experiment 2 "tip_insight_semantic" "modular" \
                MODULAR_ENABLED_PROMPTS="tip,insight" \
                MODULAR_STORAGE_TYPE="json" \
                MODULAR_RETRIEVER_TYPE="semantic"
            ;;
        3)
            run_experiment 3 "tip_insight_hybrid" "modular" \
                MODULAR_ENABLED_PROMPTS="tip,insight" \
                MODULAR_STORAGE_TYPE="json" \
                MODULAR_RETRIEVER_TYPE="hybrid" \
                MODULAR_RETRIEVER_CONFIG='{"weights":{"SemanticRetriever":0.7,"KeywordRetriever":0.3}}'
            ;;
        4)
            run_experiment 4 "tip_insight_contrastive" "modular" \
                MODULAR_ENABLED_PROMPTS="tip,insight" \
                MODULAR_STORAGE_TYPE="json" \
                MODULAR_RETRIEVER_TYPE="contrastive" \
                MODULAR_RETRIEVER_CONFIG='{"success_weight":0.6,"failure_weight":0.4,"success_k":3,"failure_k":2}'
            ;;
        5)
            run_experiment 5 "tip_insight_graph" "modular" \
                MODULAR_ENABLED_PROMPTS="tip,insight" \
                MODULAR_STORAGE_TYPE="graph" \
                MODULAR_RETRIEVER_TYPE="graph" \
                MODULAR_RETRIEVER_CONFIG='{"seed_k":3,"max_hops":1,"decay_factor":0.7}'
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

    # Skip header, read data
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
