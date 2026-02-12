#!/bin/bash

#################################################
# MemEvolve 评测配置文件
# 修改此文件来自定义评测参数
#################################################

# Conda 环境名称
export EVAL_CONDA_ENV="dl"

# 数据目录（GAIA 数据集位置）
export EVAL_DATA_DIR="./data/gaia/validation"

# 结果保存目录
export EVAL_SAVE_DIR="./evolve_demo_run"

# 评测任务数量
# - 3: 快速测试（推荐用于开发调试）
# - 10: 中等规模评测
# - 所有任务: 设置为较大数字或修改脚本
export EVAL_NUM_SAMPLE=3

# 演化轮数
# - 1: 仅运行基准评测（base round）
# - 2-5: 包含记忆系统演化
export EVAL_MAX_ROUNDS=1

# 评测模式
# - base: 仅基准评测
# - evolve: 包含演化过程
# - full: 完整流程
export EVAL_MODE="base"

# 是否清理旧结果（true/false）
# - true: 每次运行前删除旧结果
# - false: 保留旧结果（会报错如果目录已存在）
export EVAL_CLEAN_OLD_RESULTS=false

# 是否备份旧结果（true/false）
# - true: 将旧结果移动到时间戳目录
# - false: 不备份
export EVAL_BACKUP_RESULTS=true

#################################################
# 高级配置（一般不需要修改）
#################################################

# 是否显示详细日志
export EVAL_VERBOSE=false

# 超时设置（秒）
export EVAL_TIMEOUT=3600
