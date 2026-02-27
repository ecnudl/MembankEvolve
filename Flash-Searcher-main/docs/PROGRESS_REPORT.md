# PromptBasedMemoryProvider 实验进展报告

> 本文档记录了围绕 `PromptBasedMemoryProvider` 进行的所有开发、优化和实验，供新会话快速了解上下文。

---

## 一、完成的核心开发

### 1. 创建 PromptBasedMemoryProvider（新 memory provider）

**文件**: `EvolveLab/providers/prompt_based_memory_provider.py`

一个可配置的多 prompt 记忆提取与检索 provider，支持从 5 种 prompt 中选择任意组合：
- **tip** — 认知启发式规则（失败+成功均提取）
- **trajectory** — 压缩的动作-观察链
- **workflow** — 成功任务的编排逻辑（仅成功提取）
- **insight** — 失败诊断分析（仅失败提取）
- **shortcut** — 可复用宏/快捷方式

**核心流程**:
```
take_in_memory():  trajectory → Jinja2 渲染 prompt → LLM 提取 JSON → split_extraction_output() → MemoryUnit → 去重 → embedding → 持久化

provide_memory():  task query → sentence-transformers embedding → cosine similarity top-k → 格式化注入 agent 的 PlanningStep
```

**关键配置**:
- `ENABLED_PROMPTS` 环境变量控制启用哪些 prompt（逗号分隔）
- `PROMPT_STORAGE_DIR` 环境变量隔离不同实验的记忆存储
- embedding 模型: `sentence-transformers/all-MiniLM-L6-v2`（本地缓存）

### 2. 注册到框架

| 文件 | 修改内容 |
|------|---------|
| `EvolveLab/memory_types.py` | 添加 `PROMPT_BASED = "prompt_based"` 枚举 + `PROVIDER_MAPPING` |
| `EvolveLab/config.py` | 添加 `MemoryType.PROMPT_BASED` 配置块 |
| `EvolveLab/providers/__init__.py` | 导出 `PromptBasedMemoryProvider` |

### 3. 创建 shortcut 提取 prompt

**文件**: `shortcut_prompt.txt`

从 `test_memory_extraction.py` 中提取的 Shortcut Macro Extractor prompt，双模式（success: full_workflow, failure: salvaged_routine），强制参数化用 `<PLACEHOLDERS>`。

---

## 二、模型统一

将所有文件中的 LLM 模型默认值从 `qwen-plus`/`gpt-5`/`gpt-4o`/`gpt-5-mini` 统一改为 **`qwen3-max`**。

涉及 12+ 文件：`test_memory_extraction.py`, `generate_memory_trajectories.py`, `base_agent.py`, `evolve_cli.py`, `run_flash_searcher_mm_gaia.py`, `run_flash_searcher_webwalkerqa.py`, `lasj.py`, `utils.py`, `FlashOAgents/mm_tools.py`, `MemEvolve/core/memory_evolver.py`, `MemEvolve/core/auto_evolver.py`, `MemEvolve/validators/swe_agent_validator.py`, `MemEvolve/utils/trajectory_tools.py`, `MemEvolve/phases/phase_generator.py`, `MemEvolve/phases/phase_validator.py`

`.env` 文件也已更新。

---

## 三、新增工具

### GitHub API 工具

**文件**: `FlashOAgents/search_tools.py`（新增约 200 行）

| 工具 | 功能 |
|------|------|
| `GitHubSearchTool` | 通过 GitHub REST API 搜索 issues/PRs/code，支持 qualifier |
| `GitHubIssueTool` | 获取 issue 详情 + timeline 事件（label 添加时间等） |

**注册**: `base_agent.py` 中的 `MMSearchAgent` 已添加这两个工具。

**效果**: Task 9（GitHub numpy issue 查询）从 18 步/166k tokens/错误 → 7 步/48k tokens/正确。

---

## 四、Tips Prompt 优化

**文件**: `tips_prompt.txt`

- 从"两类"改为"三类"分类
- 新增 `answer_format_tips` 类别：单位转换、输出格式合规、数量-单位对齐、四舍五入/精度规则
- `EvolveLab/memory_schema.py` 的 `split_extraction_output()` 同步更新，增加 `answer_format_tips` 遍历

---

## 五、实验脚本

| 脚本 | 用途 |
|------|------|
| `run_prompt_experiments.sh` | 7 种 prompt 组合对比实验（旧版） |
| `run_all_prompt_experiments.py` | Python 版并行实验编排（旧版） |
| `run_gaia100_tip_trajectory.sh` | **最新** — 一键运行 100 task 评测（tip+trajectory），集成所有优化 |

### `run_gaia100_tip_trajectory.sh` 用法

```bash
# 默认跑前 100 个 task
./run_gaia100_tip_trajectory.sh

# 自定义范围
TASK_RANGE="1-50" ./run_gaia100_tip_trajectory.sh

# 保留旧记忆继续跑
CLEAN_MEMORY=false TASK_RANGE="51-100" ./run_gaia100_tip_trajectory.sh

# 调整并发
CONCURRENCY=3 ./run_gaia100_tip_trajectory.sh
```

---

## 六、GAIA 数据集

已通过 HuggingFace API 下载完整 GAIA 验证集：
- **165 个 task**（Level 1: 53, Level 2: 86, Level 3: 26）
- **38 个附件文件**（xlsx, pdf, mp3, png, jpg, pptx, csv, zip 等）
- 位置: `data/gaia/validation/metadata.jsonl` + 附件文件
- 旧 10 task 备份: `data/gaia/validation/metadata_10tasks_backup.jsonl`

---

## 七、实验结果（前 10 task）

### 第一轮：7 种组合对比（无 GitHub API）

| 组合 | 正确率 | 做对的 task |
|------|--------|------------|
| tip+trajectory | **60%** | 1,2,6,7,8,10 |
| workflow+shortcut | 50% | 2,6,7,8,10 |
| ALL (5种全开) | 40% | 1,2,6,7 |
| tip+workflow | 40% | 2,7,8,10 |
| trajectory+shortcut | 40% | 2,6,7,8 |
| tip+insight | 30% | 2,7,8 |
| tip+shortcut | 30% | 2,7,8 |

### 第二轮：加入 GitHub API + answer_format_tips 后

| 组合 | 正确率 | 做对的 task |
|------|--------|------------|
| tip+trajectory | **60%** | 1,2,6,7,8,**9** |
| workflow+shortcut | 50% | 2,6,7,8,**9** |

**Task 9 修复成功**（GitHub API 工具的贡献），但 Task 10 因 LLM 纯推理随机性变为错误。

### 始终失败的 task 分析

| Task | 原因 | 状态 |
|------|------|------|
| Task 3 (Nature论文数) | 搜索精度不稳定，有时能做对 | 部分改善 |
| Task 4 (Unlambda语言) | 冷门编程语言知识缺失 | 无法修复 |
| Task 5 (17 vs 17000) | "多少千小时" → agent 输出原始小时数 | 未修复（见下文） |
| Task 9 (GitHub issue) | 搜索工具无法查询 GitHub | ✅ 已修复 |
| Task 10 (乒乓球概率) | 纯 LLM 推理，不稳定 | 随机 |

---

## 八、已知问题与待解决

### Task 5 的 answer_format_tips 未生效

**根因分析**:
1. ✅ 提取阶段成功 — LLM 正确识别了 "Rounding Boundary Adherence" 等 tips
2. ✅ 存储阶段成功 — tips 已存入 memory_db.json
3. ❌ **检索阶段失败** — cosine similarity 无法将 "四舍五入/单位转换" 相关的 tip 与 "马拉松/月球距离" 的查询匹配（语义空间差距太大）
4. ❌ 注入阶段失败 — 实际注入给 agent 的是无关的物种/统计类 tips

**可能的修复方向**:
- 混合检索：cosine similarity + 关键词匹配加分
- 类型配额：top-k 中为每种 tip 类型预留位置
- 任务特征分类：判断任务是否涉及数值计算，优先检索 answer_format_tips
- 执行阶段注入：不仅在 BEGIN（规划）阶段，还在 IN（执行中）阶段注入记忆

### 实验中遇到并修复的问题

| 问题 | 修复 |
|------|------|
| `conda run --no-banner` 不支持 | 去掉该 flag |
| subprocess 输出缓冲看不到进度 | 改用 shell 直接执行 + tee 日志 |
| 并行实验共享 storage 目录导致记忆污染 | 添加 `PROMPT_STORAGE_DIR` 环境变量隔离 |
| `search_tools.py` 多余 `]` 语法错误 | 删除多余括号 |
| bash 脚本未激活 conda 环境 | 添加 `eval "$(conda shell.bash hook)" && conda activate dl` |
| 工作目录不对 | 脚本内 `cd` 到正确目录 |

---

## 九、文件变更汇总

### 新建文件
- `EvolveLab/providers/prompt_based_memory_provider.py` — 核心 provider（~400 行）
- `shortcut_prompt.txt` — shortcut 提取 prompt
- `run_prompt_experiments.sh` — 多组合实验脚本
- `run_all_prompt_experiments.py` — Python 版实验编排
- `run_gaia100_tip_trajectory.sh` — 100 task 一键评测脚本
- `EvolveLab/memory_schema.py` — MemoryUnit schema（可能是之前创建的）

### 修改文件
- `EvolveLab/memory_types.py` — 添加 PROMPT_BASED 枚举
- `EvolveLab/config.py` — 添加 PROMPT_BASED 配置
- `EvolveLab/providers/__init__.py` — 导出新 provider
- `EvolveLab/memory_schema.py` — 添加 answer_format_tips 类别
- `tips_prompt.txt` — 三类分类 + answer_format_tips
- `FlashOAgents/search_tools.py` — 添加 GitHubSearchTool + GitHubIssueTool
- `base_agent.py` — 注册 GitHub 工具到 MMSearchAgent
- `.env` — 模型改 qwen3-max + GITHUB_TOKEN
- 12+ 文件的模型默认值改为 qwen3-max
