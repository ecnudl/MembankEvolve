#!/bin/bash
# ============================================================
# GAIA 100-Task Evaluation: tip + trajectory
# ============================================================
# 一键运行脚本：使用 PromptBasedMemoryProvider (tip+trajectory)
# 评测 GAIA 验证集前 100 个 task
#
# 已集成优化：
#   - GitHub API 工具 (GitHubSearchTool + GitHubIssueTool)
#   - answer_format_tips (tips_prompt.txt 三类提取)
#   - 语义检索 (sentence-transformers/all-MiniLM-L6-v2)
#   - 记忆进化 (take_in_memory 逐任务积累)
#
# 用法:
#   chmod +x run_gaia100_tip_trajectory.sh
#   ./run_gaia100_tip_trajectory.sh
#
# 可选环境变量覆盖:
#   TASK_RANGE="1-50"       只跑前50个 (默认 1-100)
#   CONCURRENCY=3           并发数 (默认 1，建议串行以积累记忆)
#   MAX_STEPS=40            每个task最大步数 (默认 40)
#   CLEAN_MEMORY=true       运行前清除旧记忆 (默认 true)
# ============================================================

set -euo pipefail

# ---------- 配置区 ----------
TASK_RANGE="${TASK_RANGE:-1-100}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_STEPS="${MAX_STEPS:-40}"
CLEAN_MEMORY="${CLEAN_MEMORY:-true}"
ENABLED_PROMPTS="tip,trajectory"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 输出目录（带时间戳）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="gaia100_tip_trajectory_${TIMESTAMP}"
OUTPUT_DIR="./experiments/${RUN_NAME}"
STORAGE_DIR="./storage/prompt_based_tip_trajectory"
OUTFILE="${OUTPUT_DIR}/results.jsonl"

# ---------- 环境激活 ----------
eval "$(conda shell.bash hook)"
conda activate dl

# ---------- 启动信息 ----------
echo "============================================================"
echo "  GAIA 100-Task Evaluation: tip + trajectory"
echo "============================================================"
echo "  时间:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "  任务范围:   ${TASK_RANGE}"
echo "  并发数:     ${CONCURRENCY}"
echo "  最大步数:   ${MAX_STEPS}"
echo "  记忆类型:   ${ENABLED_PROMPTS}"
echo "  存储目录:   ${STORAGE_DIR}"
echo "  输出目录:   ${OUTPUT_DIR}"
echo "  清除旧记忆: ${CLEAN_MEMORY}"
echo "============================================================"

# ---------- 清理旧记忆（可选） ----------
if [ "${CLEAN_MEMORY}" = "true" ]; then
    if [ -d "${STORAGE_DIR}" ]; then
        echo "[INFO] 清除旧记忆目录: ${STORAGE_DIR}"
        rm -rf "${STORAGE_DIR}"
    fi
fi

# ---------- 创建目录 ----------
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${STORAGE_DIR}"

# ---------- 检查数据集 ----------
INFILE="./data/gaia/validation/metadata.jsonl"
if [ ! -f "${INFILE}" ]; then
    echo "[ERROR] 数据集不存在: ${INFILE}"
    exit 1
fi
TOTAL_TASKS=$(wc -l < "${INFILE}")
echo "[INFO] 数据集共 ${TOTAL_TASKS} 个任务，本次运行范围: ${TASK_RANGE}"

# ---------- 检查关键环境变量 ----------
if [ -z "${OPENAI_API_KEY:-}" ]; then
    source .env 2>/dev/null || true
fi
echo "[INFO] Model: ${DEFAULT_MODEL:-qwen3-max}"
echo "[INFO] GitHub Token: ${GITHUB_TOKEN:+已配置}"
echo ""

# ---------- 运行评测 ----------
echo "[INFO] 开始评测... (日志同时输出到 ${OUTPUT_DIR}/run.log)"
echo ""

ENABLED_PROMPTS="${ENABLED_PROMPTS}" \
PROMPT_STORAGE_DIR="${STORAGE_DIR}" \
python run_flash_searcher_mm_gaia.py \
    --infile "${INFILE}" \
    --outfile "${OUTFILE}" \
    --task_indices "${TASK_RANGE}" \
    --memory_provider prompt_based \
    --enable_memory_evolution \
    --concurrency "${CONCURRENCY}" \
    --max_steps "${MAX_STEPS}" \
    --direct_output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

# ---------- 运行完成，生成统计 ----------
echo ""
echo "============================================================"
echo "  评测完成"
echo "============================================================"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  结果目录: ${OUTPUT_DIR}"
echo ""

# 统计正确率
if [ -f "${OUTPUT_DIR}/report.txt" ]; then
    echo "--- 评测报告 ---"
    cat "${OUTPUT_DIR}/report.txt"
else
    # 手动统计
    CORRECT=$(grep -rl '"judgement".*[Cc]orrect' "${OUTPUT_DIR}"/*.json 2>/dev/null | wc -l || echo 0)
    TOTAL=$(ls "${OUTPUT_DIR}"/*.json 2>/dev/null | wc -l || echo 0)
    echo "  完成任务: ${TOTAL}"
    echo "  正确数:   ${CORRECT}"
    if [ "${TOTAL}" -gt 0 ]; then
        echo "  正确率:   $(echo "scale=1; ${CORRECT} * 100 / ${TOTAL}" | bc)%"
    fi
fi

echo ""
echo "  记忆文件: ${STORAGE_DIR}/memory_db.json"
if [ -f "${STORAGE_DIR}/memory_db.json" ]; then
    MEMORY_COUNT=$(python3 -c "import json; print(len(json.load(open('${STORAGE_DIR}/memory_db.json'))))" 2>/dev/null || echo "?")
    echo "  记忆单元数: ${MEMORY_COUNT}"
fi
echo "============================================================"
