请帮我配置并运行当前仓库中 "MemEvolve" 记忆进化框架的 Demo。

由于我是中文用户，请**全程使用中文**与我进行交互和解释。

### 核心任务目标
1.  跑通 MemEvolve 的 `auto-evolve` 流程（使用 Qwen 模型）。
2.  **关键要求**：确保运行产生的**记忆轨迹 (Memory Trajectory)** 被正确保存。运行结束后，请你帮我找到存储了 `agent_trajectory` 字段的 JSON 日志文件，并告诉我它的具体路径。我需要用这个文件来可视化分析记忆的工作过程。

### 配置要求与限制
1.  **目标数据集**: GAIA (Validation set)。
2.  **LLM 提供商**: 使用 **Qwen (通义千问)** 替代 OpenAI。
    * Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
    * 模型名称: `qwen-max` (用于替代 GPT-4)。
    * API Key: 我稍后会提供 `DASHSCOPE_API_KEY`，请将其填入 `.env` 文件中的 `OPENAI_API_KEY` 字段（因为代码复用了 OpenAI 客户端）。
3.  **搜索工具**: 使用 `Serper` (我会提供 Key)。
4.  **网页浏览**: 使用 `crawl4ai` (本地浏览器) 替代 Jina。

---

请按照以下步骤顺序执行操作：

### 第一步：环境配置 (.env)
请在根目录下创建一个 `.env` 文件，写入以下内容。**在写入前，请询问我具体的 API Key**。

```bash
# === 核心 LLM 配置 (使用 Qwen 兼容接口) ===
DEFAULT_MODEL="qwen-max"
OPENAI_BASE_URL="[https://dashscope.aliyuncs.com/compatible-mode/v1](https://dashscope.aliyuncs.com/compatible-mode/v1)"
OPENAI_API_KEY="sk-..."  # 占位符，请询问用户获取 Qwen/Dashscope Key

# === 进化模块专用模型配置 ===
ANALYSIS_MODEL="qwen-max"
GENERATION_MODEL="qwen-max"

# === 搜索与爬虫配置 ===
SERPER_API_KEY="..."     # 占位符，请询问用户获取 Serper Key
WEB_ACCESS_PROVIDER="crawl4ai"
### 第二步：依赖安装

1. 运行 `pip install -r requirements.txt` 安装 Python 依赖。
2. **重要**：因为检测到使用 `crawl4ai`，请务必在 pip 安装完成后，运行 `playwright install` 以安装必要的浏览器内核，否则爬虫会失败。

### 第三步：数据检查

检查 `./data/gaia` 目录是否存在。

- 如果目录为空或缺少 `validation-v0.1.jsonl` 文件，请尝试从 HuggingFace 下载 GAIA 验证集，或者创建一个包含 3-5 条测试数据的 dummy 文件 (JSONL格式)，以确保流程可以运行。

### 第四步：运行进化 Demo

使用 `evolve_cli.py` 的 `auto-evolve` 模式运行一个快速验证 Demo。

**请使用以下命令（已包含保存路径配置）：**

Bash

```
python evolve_cli.py auto-evolve gaia \
  --num-rounds 1 \
  --num-systems 1 \
  --task-batch-x 3 \
  --top-t 1 \
  --extra-sample-y 2 \
  --work-dir ./evolve_demo_run
```

### 第五步：验证与提取轨迹 (Trajectory)

运行完成后，请务必执行以下检查并向我汇报：

1. 进入 `./evolve_demo_run` 目录（或其生成的子目录），找到生成的任务日志文件（通常是 `.json` 格式，文件名可能是数字索引）。
2. 读取其中一个日志文件，**验证其中是否包含 `agent_trajectory` 字段**。这个字段应该包含 `think` (思考)、`tool_calls` (工具调用) 和 `memory_guidance` (记忆指导) 等详细信息。
3. **最终输出**：请明确告诉我：“您的记忆轨迹文件保存在路径：[具体文件的绝对路径]”，以便我直接去读取进行可视化分析。