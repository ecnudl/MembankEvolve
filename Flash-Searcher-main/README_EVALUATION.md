# MemEvolve 评测脚本使用指南

本指南说明如何使用自动化评测脚本来运行 MemEvolve 记忆轨迹分析。

## 快速开始

### 方法 1: 使用默认配置（推荐）

```bash
cd /home/MemEvolve/Flash-Searcher-main
./run_evaluation.sh
```

### 方法 2: 使用自定义配置

1. 修改配置文件：
```bash
nano evaluation_config.sh
```

2. 加载配置并运行：
```bash
source evaluation_config.sh
./run_evaluation.sh
```

### 方法 3: 临时修改参数

```bash
# 修改脚本中的配置参数部分
nano run_evaluation.sh

# 找到以下部分并修改：
# NUM_SAMPLE=3           # 改为你想要的任务数量
# MAX_ROUNDS=1           # 改为你想要的演化轮数
# CLEAN_OLD_RESULTS=false  # 是否清理旧结果

# 然后运行
./run_evaluation.sh
```

## 配置参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_SAMPLE` | 3 | 评测任务数量（建议3-10） |
| `MAX_ROUNDS` | 1 | 演化轮数（1=仅基准，>1=包含演化） |
| `MODE` | base | 评测模式（base/evolve/full） |
| `DATA_DIR` | ./data/gaia/validation | GAIA数据集路径 |
| `SAVE_DIR` | ./evolve_demo_run | 结果保存路径 |

### 高级选项

- `CLEAN_OLD_RESULTS=true`: 运行前清理旧结果
- `BACKUP_RESULTS=true`: 将旧结果备份到时间戳目录

## 评测流程

脚本会自动执行以下步骤：

1. ✓ 检查 Conda 环境（dl）
2. ✓ 验证数据目录和配置文件
3. ✓ 清理/备份旧结果（如果启用）
4. ✓ 运行评测（调用 evolve_cli.py）
5. ✓ 分析结果并生成统计信息
6. ✓ 显示结果文件位置

## 输出文件

评测完成后，会在 `evolve_demo_run/round_00/` 目录下生成：

### 1. 任务轨迹文件
- **位置**: `base_logs/1.json`, `base_logs/2.json`, ...
- **内容**: 每个任务的完整执行轨迹
  - `agent_trajectory`: Agent思考和工具调用记录
  - `memory_guidance`: 记忆系统的引导信息
  - `final_answer`: 最终答案

### 2. 评测结果文件
- **位置**: `result.jsonl`
- **内容**: 每个任务的答案和评判结果
  ```json
  {
    "task_id": "gaia-001",
    "answer": "1940",
    "judgement": "correct"
  }
  ```

### 3. 分析报告（如果启用分析）
- **位置**: `analysis_report.json`
- **内容**: 记忆操作的详细分析
  - PROVIDE 操作分析
  - TAKE-IN 操作分析
  - MANAGEMENT 操作分析

## 结果查看命令

```bash
# 查看任务轨迹（格式化JSON）
cat evolve_demo_run/round_00/base_logs/1.json | jq .

# 查看评测结果
cat evolve_demo_run/round_00/result.jsonl | jq .

# 查看分析报告
cat evolve_demo_run/round_00/analysis_report.json | jq .

# 统计任务正确率
grep -o '"judgement": "[^"]*"' evolve_demo_run/round_00/result.jsonl | sort | uniq -c
```

## 常见问题

### Q1: 如何评测更多任务？
修改 `NUM_SAMPLE` 参数：
```bash
# 在 run_evaluation.sh 中修改
NUM_SAMPLE=10  # 评测10个任务
```

### Q2: 如何运行完整的演化流程？
修改以下参数：
```bash
MAX_ROUNDS=3  # 运行3轮演化
MODE="full"   # 完整模式
```

### Q3: 旧结果如何处理？
- 自动清理: `CLEAN_OLD_RESULTS=true`
- 自动备份: `BACKUP_RESULTS=true`
- 手动清理: `rm -rf evolve_demo_run`

### Q4: 如何使用不同的GAIA数据集？
修改 `DATA_DIR` 参数：
```bash
DATA_DIR="./data/gaia/test"  # 使用测试集
```

### Q5: 评测失败怎么办？
检查以下项目：
1. Conda 环境是否正确激活
2. `.env` 文件是否配置完整
3. API keys 是否有效
4. 数据文件是否存在

## 性能建议

- **快速测试**: `NUM_SAMPLE=3`, `MAX_ROUNDS=1` (~5分钟)
- **标准评测**: `NUM_SAMPLE=10`, `MAX_ROUNDS=1` (~15分钟)
- **完整演化**: `NUM_SAMPLE=10`, `MAX_ROUNDS=3` (~1小时)

## 自定义数据集

如果要使用自己的数据集：

1. 创建数据目录：
```bash
mkdir -p data/custom_dataset
```

2. 准备 `metadata.jsonl` 文件（GAIA格式）：
```json
{"task_id": "custom-001", "Question": "Your question here", "Final answer": "Expected answer", "Level": 1, "file_name": ""}
```

3. 修改脚本中的 `DATA_DIR`：
```bash
DATA_DIR="./data/custom_dataset"
```

## 技术支持

如遇问题，请检查：
- 日志输出中的错误信息
- `.env` 文件配置
- API 配额和网络连接
- 数据文件格式

---

**注意**: 评测过程会调用 LLM API，请确保 API key 有足够的配额。
