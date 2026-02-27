#!/bin/bash
# PromptBasedMemoryProvider Experiment Runner
# Runs 7 prompt combinations against GAIA benchmark to compare
# which prompt subsets are most effective for memory extraction.
#
# Usage:
#   chmod +x run_prompt_experiments.sh
#   ./run_prompt_experiments.sh
#
# Prerequisites:
#   - Set OPENAI_API_KEY and OPENAI_API_BASE in .env or environment
#   - Ensure GAIA dataset is available

set -euo pipefail

COMBINATIONS=(
    "tip,workflow"
    "tip,trajectory"
    "tip,insight"
    "workflow,shortcut"
    "trajectory,shortcut"
    "tip,shortcut"
    "insight,tip,trajectory,workflow,shortcut"
)

TASK_INDICES="${TASK_INDICES:-1-20}"
CONCURRENCY="${CONCURRENCY:-1}"
EXPERIMENTS_DIR="./experiments"

echo "=============================="
echo "Prompt-Based Memory Experiments"
echo "=============================="
echo "Task indices: ${TASK_INDICES}"
echo "Concurrency:  ${CONCURRENCY}"
echo "Combinations: ${#COMBINATIONS[@]}"
echo ""

for combo in "${COMBINATIONS[@]}"; do
    name=$(echo "$combo" | tr ',' '_')
    out_dir="${EXPERIMENTS_DIR}/prompt_${name}"
    outfile="${out_dir}/results.jsonl"

    echo "--- Starting: ${combo} ---"

    # Clean previous storage for this run
    rm -rf ./storage/prompt_based/

    mkdir -p "${out_dir}"

    ENABLED_PROMPTS="$combo" python run_flash_searcher_mm_gaia.py \
        --memory_provider prompt_based \
        --task_indices "$TASK_INDICES" \
        --concurrency "$CONCURRENCY" \
        --direct_output_dir "$out_dir" \
        --outfile "$outfile"

    echo "=== Completed: ${name} ==="
    echo ""
done

echo "=============================="
echo "All experiments completed."
echo "Results in: ${EXPERIMENTS_DIR}/"
echo "=============================="
