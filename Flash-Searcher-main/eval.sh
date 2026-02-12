#!/bin/bash

#################################################
# MemEvolve 快速评测启动脚本
# 使用方法: ./eval.sh
#################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载配置文件（如果存在）
if [ -f "evaluation_config.sh" ]; then
    echo "📝 加载配置文件: evaluation_config.sh"
    source evaluation_config.sh
fi

# 运行评测脚本
./run_evaluation.sh
