#!/usr/bin/env python
"""
Run all 7 prompt combination experiments sequentially.
Each experiment runs 10 GAIA tasks with a different enabled_prompts config.
"""
import os
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path

COMBINATIONS = [
    "tip,workflow",
    "tip,trajectory",
    "tip,insight",
    "workflow,shortcut",
    "trajectory,shortcut",
    "tip,shortcut",
    "insight,tip,trajectory,workflow,shortcut",
]

EXPERIMENTS_DIR = Path("./experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

INFILE = "./data/gaia/validation/metadata.jsonl"
TASK_INDICES = "1-10"

results_summary = []

for i, combo in enumerate(COMBINATIONS, 1):
    name = combo.replace(",", "_")
    out_dir = EXPERIMENTS_DIR / f"prompt_{name}"
    outfile = out_dir / "results.jsonl"

    print(f"\n{'='*60}")
    print(f"[{i}/{len(COMBINATIONS)}] Running: {combo}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    # Clean previous storage
    storage_dir = Path("./storage/prompt_based")
    if storage_dir.exists():
        shutil.rmtree(storage_dir)

    # Clean previous output
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ENABLED_PROMPTS"] = combo

    cmd = [
        sys.executable, "run_flash_searcher_mm_gaia.py",
        "--infile", INFILE,
        "--outfile", str(outfile),
        "--task_indices", TASK_INDICES,
        "--concurrency", "1",
        "--max_steps", "40",
        "--memory_provider", "prompt_based",
        "--enable_memory_evolution",
        "--direct_output_dir", str(out_dir),
    ]

    start = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=False, text=True)
    elapsed = time.time() - start

    # Collect results
    correct = 0
    total = 0
    if outfile.exists():
        with open(outfile, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    total += 1
                    judgement = r.get("judgement", "").strip().lower()
                    if judgement == "correct":
                        correct += 1
                except json.JSONDecodeError:
                    pass

    # Also check individual json files
    if total == 0:
        for jf in sorted(out_dir.glob("*.json")):
            if jf.name == "experiment_summary.json":
                continue
            try:
                with open(jf) as f:
                    r = json.load(f)
                total += 1
                judgement = r.get("judgement", "").strip().lower()
                if judgement == "correct":
                    correct += 1
            except:
                pass

    accuracy = correct / total * 100 if total > 0 else 0
    entry = {
        "combination": combo,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 1),
        "returncode": proc.returncode,
    }
    results_summary.append(entry)

    print(f"\n>>> {combo}: {correct}/{total} correct ({accuracy:.1f}%) in {elapsed:.0f}s\n")

# Final summary
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
results_summary.sort(key=lambda x: x["accuracy"], reverse=True)
for r in results_summary:
    star = " *** BEST ***" if r == results_summary[0] else ""
    print(f"  {r['combination']:50s} {r['correct']}/{r['total']} = {r['accuracy']:5.1f}%  ({r['elapsed_seconds']:.0f}s){star}")

# Save summary
summary_path = EXPERIMENTS_DIR / "summary.json"
with open(summary_path, "w") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)
print(f"\nSummary saved to: {summary_path}")

best = results_summary[0]
print(f"\nBest combination: {best['combination']} ({best['accuracy']:.1f}%)")
