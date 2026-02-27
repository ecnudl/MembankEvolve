#!/bin/bash
# Run remaining 6 experiments in parallel
# Must be sourced within conda dl environment

eval "$(conda shell.bash hook)"
conda activate dl

COMBOS=("tip,trajectory" "tip,insight" "workflow,shortcut" "trajectory,shortcut" "tip,shortcut" "insight,tip,trajectory,workflow,shortcut")
PIDS=()

for combo in "${COMBOS[@]}"; do
    name=$(echo "$combo" | tr ',' '_')
    out_dir="./experiments/prompt_${name}"
    storage_dir="./storage/prompt_based_${name}"

    rm -rf "$out_dir" "$storage_dir"
    mkdir -p "$out_dir"

    echo "[$(date '+%H:%M:%S')] Starting: $combo"

    ENABLED_PROMPTS="$combo" \
    PROMPT_STORAGE_DIR="$storage_dir" \
    python run_flash_searcher_mm_gaia.py \
        --infile ./data/gaia/validation/metadata.jsonl \
        --outfile "${out_dir}/results.jsonl" \
        --task_indices "1-10" \
        --concurrency 1 \
        --max_steps 40 \
        --memory_provider prompt_based \
        --enable_memory_evolution \
        --direct_output_dir "$out_dir" \
        > "${out_dir}/run.log" 2>&1 &

    PIDS+=($!)
    echo "  PID: $! -> $out_dir"
done

echo ""
echo "[$(date '+%H:%M:%S')] All 6 experiments launched. Waiting..."

for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    rc=$?
    combo=${COMBOS[$i]}
    name=$(echo "$combo" | tr ',' '_')
    out_dir="./experiments/prompt_${name}"

    correct=$(python3 -c "
import json, glob
correct=0; total=0
for f in sorted(glob.glob('${out_dir}/*.json')):
    if 'summary' in f: continue
    try:
        d=json.load(open(f))
        total+=1
        if d.get('judgement','').strip().lower()=='correct': correct+=1
    except: pass
print(f'{correct}/{total}')
" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] Done: ${combo} -> ${correct} (exit: ${rc})"
done

echo ""
echo "[$(date '+%H:%M:%S')] All experiments completed!"
