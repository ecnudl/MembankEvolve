#!/bin/bash

#################################################
# MemEvolve GAIA Evaluation Script
# 用于运行 MemEvolve 记忆轨迹分析评测
#################################################

set -e  # 遇到错误立即退出

# ========== 配置参数 ==========
CONDA_ENV="dl"
DATA_DIR="./data/gaia/validation"
SAVE_DIR="./evolve_demo_run"
NUM_SAMPLE=3           # 评测任务数量
MAX_ROUNDS=1           # 演化轮数（1=仅基准评测，>1=包含演化）
MODE="base"            # 模式: base, evolve, full

# 可选：是否清理旧结果（true/false）
CLEAN_OLD_RESULTS=false

# 可选：是否备份结果到时间戳目录（true/false）
BACKUP_RESULTS=false

# ========== 颜色输出 ==========
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========== 函数定义 ==========
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# ========== 主流程 ==========
print_header "MemEvolve GAIA 评测开始"

# 1. 检查conda环境
print_info "检查 Conda 环境: $CONDA_ENV"
if ! conda env list | grep -q "^$CONDA_ENV "; then
    print_error "Conda 环境 '$CONDA_ENV' 不存在"
    print_info "请先创建环境: conda create -n $CONDA_ENV python=3.10"
    exit 1
fi
print_success "Conda 环境检查通过"

# 2. 检查数据目录
print_info "检查数据目录: $DATA_DIR"
if [ ! -f "$DATA_DIR/metadata.jsonl" ]; then
    print_error "数据文件不存在: $DATA_DIR/metadata.jsonl"
    exit 1
fi
print_success "数据目录检查通过"

# 3. 检查.env配置
print_info "检查环境配置文件"
if [ ! -f ".env" ]; then
    print_error ".env 文件不存在，请先配置 API keys"
    exit 1
fi
print_success "环境配置检查通过"

# 4. 清理旧结果（可选）
if [ "$CLEAN_OLD_RESULTS" = true ]; then
    print_info "清理旧的评测结果..."
    if [ -d "$SAVE_DIR" ]; then
        rm -rf "$SAVE_DIR"
        print_success "旧结果已清理"
    fi
fi

# 5. 备份旧结果（可选）
if [ "$BACKUP_RESULTS" = true ] && [ -d "$SAVE_DIR" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_DIR="${SAVE_DIR}_backup_${TIMESTAMP}"
    print_info "备份旧结果到: $BACKUP_DIR"
    mv "$SAVE_DIR" "$BACKUP_DIR"
    print_success "备份完成"
fi

# 6. 显示评测配置
print_header "评测配置"
echo "数据目录: $DATA_DIR"
echo "保存目录: $SAVE_DIR"
echo "任务数量: $NUM_SAMPLE"
echo "演化轮数: $MAX_ROUNDS"
echo "评测模式: $MODE"
echo ""

# 7. 运行评测
print_header "开始运行评测"
print_info "这可能需要几分钟时间，请耐心等待..."
echo ""

# 使用 conda run 执行评测
conda run -n "$CONDA_ENV" python evolve_cli.py \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --num_sample "$NUM_SAMPLE" \
    --max_rounds "$MAX_ROUNDS" \
    --mode "$MODE"

# 8. 检查结果
print_header "评测完成，分析结果"

if [ ! -d "$SAVE_DIR/round_00/base_logs" ]; then
    print_error "未找到评测结果目录"
    exit 1
fi

# 统计任务文件数量
TASK_COUNT=$(ls -1 "$SAVE_DIR/round_00/base_logs/"*.json 2>/dev/null | wc -l)
print_success "生成任务轨迹文件: $TASK_COUNT 个"

# 统计任务正确率
if [ -f "$SAVE_DIR/round_00/result.jsonl" ]; then
    CORRECT_COUNT=$(grep -o '"judgement": "correct"' "$SAVE_DIR/round_00/result.jsonl" | wc -l)
    TOTAL_COUNT=$(wc -l < "$SAVE_DIR/round_00/result.jsonl")
    if [ "$TOTAL_COUNT" -gt 0 ]; then
        ACCURACY=$(echo "scale=2; $CORRECT_COUNT * 100 / $TOTAL_COUNT" | bc)
        print_success "任务正确率: $CORRECT_COUNT/$TOTAL_COUNT = ${ACCURACY}%"
    fi
fi

# 检查分析报告
if [ -f "$SAVE_DIR/round_00/analysis_report.json" ]; then
    REPORT_SIZE=$(du -h "$SAVE_DIR/round_00/analysis_report.json" | cut -f1)
    print_success "分析报告已生成: analysis_report.json ($REPORT_SIZE)"
else
    print_info "未找到分析报告（可能未执行分析阶段）"
fi

# 9. 显示结果位置
print_header "结果文件位置"
echo "📁 基础轨迹日志: $SAVE_DIR/round_00/base_logs/"
echo "   - 每个任务的完整执行轨迹（包含 agent_trajectory）"
echo ""
echo "📊 评测结果: $SAVE_DIR/round_00/result.jsonl"
echo "   - 每个任务的答案和评判结果"
echo ""
if [ -f "$SAVE_DIR/round_00/analysis_report.json" ]; then
    echo "📈 分析报告: $SAVE_DIR/round_00/analysis_report.json"
    echo "   - 记忆操作的详细分析（PROVIDE, TAKE-IN, MANAGEMENT）"
    echo ""
fi

# 10. 显示快速查看命令
print_header "快速查看结果命令"
echo "# 查看任务轨迹（JSON格式化）"
echo "cat $SAVE_DIR/round_00/base_logs/1.json | jq ."
echo ""
echo "# 查看评测结果"
echo "cat $SAVE_DIR/round_00/result.jsonl | jq ."
echo ""
if [ -f "$SAVE_DIR/round_00/analysis_report.json" ]; then
    echo "# 查看分析报告"
    echo "cat $SAVE_DIR/round_00/analysis_report.json | jq ."
    echo ""
fi
echo "# 查看任务正确率统计"
echo "grep -o '\"judgement\": \"[^\"]*\"' $SAVE_DIR/round_00/result.jsonl | sort | uniq -c"
echo ""

print_header "评测流程全部完成"
print_success "所有结果已保存到: $SAVE_DIR"
