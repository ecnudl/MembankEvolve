#!/usr/bin/env python
# coding=utf-8
"""
Memory Extraction Prompt Tester

Tests your custom memory extraction prompt against trajectory data.
Supports Jinja2 templating for conditional prompt logic.

Usage:
    python test_memory_extraction.py
    python test_memory_extraction.py --trajectories_dir ./memory_trajectories
    python test_memory_extraction.py --prompt_file my_prompt.txt
    python test_memory_extraction.py --task_ids "gaia-001,gaia-006"
"""

import os
import json
import re
import logging
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override=True)

import jinja2
from FlashOAgents import OpenAIServerModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================================
# >>>  在这里填写你的记忆抽取 Prompt（支持 Jinja2 语法）<<<
#
# 可用的模板变量:
#   {{ task_query }}              - 任务问题
#   {{ is_success }}              - 是否成功 (bool: True/False)
#   {{ raw_trajectory }}          - 格式化后的完整执行轨迹
#   {{ final_result }}            - agent的最终答案
#   {{ golden_answer }}           - 标准答案
#   {{ task_id }}                 - 任务ID
#   {{ task_order }}              - 任务执行顺序
#   {{ memory_guidance }}         - 本次任务收到的记忆指导
#   {{ failure_reason }}          - 失败原因（成功时为None）
#   {{ reference_trajectory }}    - 前一个任务的轨迹（用于对比分析，第一个任务为None）
#   {{ memory_count_before }}     - 执行前记忆库中的条目数
#
# 支持 Jinja2 完整语法: {% if %}, {% for %}, {{ var | default() }} 等
# ============================================================================

EXTRACTION_PROMPT = """## 1. Role Definition
You are the **Shortcut Macro Extractor**. You extract **ready-to-run, reusable** macro actions (scripts/commands/templates) with explicit environment dependencies. You focus on generalizable execution patterns, NOT task-specific replay scripts.

## 2. Input Context
- Task Query: {{ task_query }}
- Success Flag: {{ is_success }}
- Raw Trajectory Log: {{ raw_trajectory }}
- Final Result: {{ final_result }}
- Failure Reason (optional): {{ failure_reason | default("None", true) }}
- Reference Trajectory (optional): {{ reference_trajectory | default("None", true) }}

## 3. Extraction Rules
1. Dual-Mode:
   {% if is_success %}
   - Extract 1–3 macros from SUCCESS.
   {% else %}
   - Extract 1–3 salvaged macros from FAILURE and fill `failure_diagnosis` (non-null).
   {% endif %}

2. Current-Trace Only:
   - ONLY use evidence from `raw_trajectory`. Ignore `reference_trajectory` for generation.

3. No Invented Tools (Hard):
   - `executable_payload` may ONLY use tools/interfaces that **appear in `raw_trajectory`**.
   - Do NOT introduce curl/jq/bs4/Wikipedia API/Google CSE/regex scrapers unless they were actually executed.

4. Interface Fidelity (Hard):
   - If the trace uses tool-calls (e.g., `web_search`, `crawl_page`, `final_answer`), express payload in the same tool-call style.
   - Only output shell/Python code if shell/Python was executed in the trace.

5. Mandatory Parameterization (Hard):
   - Replace task-specific values with typed <PLACEHOLDERS> (numbers/dates/names/URLs/paths/results).
   - Never hardcode intermediate/final results; use `<COMPUTED_VALUE>`.
   - If a placeholder is extracted from a previous step, the `intent` must state the source (e.g., “extract <ARTICLE_COUNT> from crawl_page observation”).

6. Macro Must Be Replayable (Hard):
   - Include all required steps that were used in the trace (e.g., if URL came from `web_search`, include that step).
   - If the solution requires arithmetic:
     - Use a computation tool only if it appears in the trace.
     - Otherwise include an explicit formula step (e.g., `COMPUTE: <COMPUTED_VALUE>=ceil(<ARTICLE_COUNT>*<P_VALUE>)`) and set an assumption that the runtime/agent computes it.

7. Assumptions = Evidence-Based:
   - List only what’s evidenced in `raw_trajectory` (plus generic “network access” if needed). No speculative deps.

8. Diversity:
   - Each macro covers a distinct sub-capability; prefer 1–2 high-quality macros over redundancy.

{% if not is_success %}
9. Salvage Safety:
   - Do not preserve the exact bug that caused failure; fix it or omit the macro.
   - `failure_diagnosis` must be specific and evidence-grounded.
{% endif %}

X. Strict JSON Syntax (Hard):
   - Output strictly valid JSON. NO trailing comma before `}` or `]`.

## 4. Output Schema (JSON)
```json
[
  {
    "name": "snake_case_macro_name",
    "description": "1-2 sentences describing the GENERAL reusable pattern.",
    "precondition": "General trigger condition (no task-specific entities).",
    "extraction_type": "{% if is_success %}full_workflow{% else %}salvaged_routine{% endif %}",
    "assumptions": [
      "Only items evidenced in raw_trajectory (plus 'network access' if needed)"
    ],
    "failure_diagnosis": {% if not is_success %}{
      "root_cause": "1-sentence, evidence-grounded",
      "correction_hint": "1-sentence, minimal"
    }{% else %}null{% endif %},
    "action_sequence": [
      {
        "step": 1,
        "intent": "One-line intent; state where placeholders come from if extracted.",
        "executable_payload": "Tool-call style or trace-evidenced code with <PLACEHOLDERS>."
      }
    ]
  }
]
```"""


# ============================================================================
# 以下是脚本逻辑
# ============================================================================


def format_trajectory_text(trajectory_data: dict) -> str:
    """将轨迹JSON格式化为可读文本"""
    parts = []
    agent_trajectory = trajectory_data.get("agent_trajectory", [])

    for i, step in enumerate(agent_trajectory):
        step_name = step.get("name", "unknown")

        if step_name == "plan":
            plan_text = step.get("value", "")
            if len(plan_text) > 2000:
                plan_text = plan_text[:2000] + "\n... [truncated]"
            parts.append(f"[Step {i} - Plan]\n{plan_text}")

        elif step_name == "action":
            tool_calls = step.get("tool_calls", [])
            obs = step.get("obs", "")
            think = step.get("think", "")

            tc_texts = []
            for tc in tool_calls:
                tc_name = tc.get("name", "unknown")
                tc_args = tc.get("arguments", {})
                if isinstance(tc_args, dict):
                    tc_args_str = json.dumps(tc_args, ensure_ascii=False)
                else:
                    tc_args_str = str(tc_args)
                tc_texts.append(f"  - {tc_name}({tc_args_str})")

            action_text = f"[Step {i} - Action]"
            if think:
                think_display = think[:500] + "..." if len(think) > 500 else think
                action_text += f"\nThinking: {think_display}"
            if tc_texts:
                action_text += f"\nTool Calls:\n" + "\n".join(tc_texts)
            if obs:
                obs_display = obs[:1500] + "\n... [truncated]" if len(obs) > 1500 else obs
                action_text += f"\nObservations:\n{obs_display}"

            parts.append(action_text)

        elif step_name == "summary":
            summary_text = step.get("value", "")
            if len(summary_text) > 1000:
                summary_text = summary_text[:1000] + "\n... [truncated]"
            parts.append(f"[Step {i} - Summary]\n{summary_text}")

    return "\n\n".join(parts)


def extract_memory_guidance(trajectory_data: dict) -> str:
    """提取轨迹中的memory_guidance内容"""
    agent_trajectory = trajectory_data.get("agent_trajectory", [])
    guidances = []
    for step in agent_trajectory:
        mg = step.get("memory_guidance")
        if mg:
            guidances.append(mg)
    return "\n---\n".join(guidances) if guidances else ""


def derive_failure_reason(trajectory_data: dict) -> str:
    """从轨迹数据推导失败原因"""
    if trajectory_data.get("status") == "error":
        return trajectory_data.get("error", "Unknown error")

    judgement = trajectory_data.get("judgement", "")
    if judgement and judgement.strip().lower() != "correct":
        agent_result = str(trajectory_data.get("agent_result", ""))
        golden = str(trajectory_data.get("golden_answer", ""))
        return f"Answer mismatch: agent answered '{agent_result}', expected '{golden}'"

    return None


def build_template_context(trajectory_data: dict, prev_trajectory_data: dict = None) -> dict:
    """
    构建Jinja2模板的上下文变量。
    将轨迹JSON中的字段映射为prompt模板需要的变量。
    """
    # 判断是否成功
    judgement = trajectory_data.get("judgement", "")
    is_success = judgement.strip().lower() == "correct" if judgement else False

    # 如果status是error，也算失败
    if trajectory_data.get("status") == "error":
        is_success = False

    # 格式化轨迹
    raw_trajectory = format_trajectory_text(trajectory_data)
    memory_guidance = extract_memory_guidance(trajectory_data)
    failure_reason = derive_failure_reason(trajectory_data)

    # 格式化参考轨迹（前一个任务的轨迹，用于对比分析）
    reference_trajectory = None
    if prev_trajectory_data:
        reference_trajectory = format_trajectory_text(prev_trajectory_data)

    return {
        "task_query": trajectory_data.get("question", ""),
        "is_success": is_success,
        "raw_trajectory": raw_trajectory,
        "final_result": str(trajectory_data.get("agent_result", "")),
        "golden_answer": str(trajectory_data.get("golden_answer", "")),
        "task_id": trajectory_data.get("task_id", ""),
        "task_order": trajectory_data.get("task_order", 0),
        "memory_guidance": memory_guidance if memory_guidance else None,
        "failure_reason": failure_reason,
        "reference_trajectory": reference_trajectory,
        "memory_count_before": trajectory_data.get("memory_count_before", 0),
    }


def render_prompt(template_str: str, context: dict) -> str:
    """用Jinja2渲染prompt模板"""
    env = jinja2.Environment(undefined=jinja2.Undefined)
    template = env.from_string(template_str)
    return template.render(**context)


def call_llm(prompt: str, model_config: dict) -> str:
    """调用LLM获取响应"""
    model = OpenAIServerModel(**model_config)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    response = model(messages)
    return response.content if hasattr(response, 'content') else str(response)


def parse_json_from_response(response_text: str) -> dict:
    """从LLM响应中提取JSON对象"""
    # 尝试1: 直接解析
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # 尝试2: 提取 ```json ... ``` 代码块
    json_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试3: 提取最外层的 [ ... ] (数组) 或 { ... } (对象)
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
        json_match = re.search(pattern, response_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                continue

    logger.warning(f"Failed to parse JSON from response: {response_text[:300]}...")
    return {"_raw_response": response_text, "_parse_error": True}


def main():
    parser = argparse.ArgumentParser(description='Test Memory Extraction Prompt')
    parser.add_argument('--trajectories_dir', type=str,
                        default="./memory_trajectories",
                        help='Directory containing trajectory JSON files')
    parser.add_argument('--output_file', type=str,
                        default="./memory_extraction_results.json",
                        help='Output file for extraction results')
    parser.add_argument('--prompt_file', type=str, default=None,
                        help='Load prompt from a text file (supports Jinja2)')
    parser.add_argument('--task_ids', type=str, default=None,
                        help='Only process specific task IDs (e.g., "gaia-001,gaia-006")')
    parser.add_argument('--skip_llm_on_success', action='store_true', default=False,
                        help='Skip LLM call for successful tasks (directly return skipped output)')
    parser.add_argument('--skip_llm_on_failure', action='store_true', default=False,
                        help='Skip LLM call for failed tasks (directly return skipped output)')
    args = parser.parse_args()

    # 加载prompt模板
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            logger.error(f"Prompt file not found: {prompt_path}")
            return
        prompt_template = prompt_path.read_text(encoding='utf-8')
        logger.info(f"Loaded prompt from: {prompt_path}")
    else:
        prompt_template = EXTRACTION_PROMPT
        logger.info("Using built-in EXTRACTION_PROMPT")

    # 加载轨迹文件
    traj_dir = Path(args.trajectories_dir)
    if not traj_dir.exists():
        logger.error(f"Trajectories directory not found: {traj_dir}")
        return

    traj_files = sorted([
        f for f in traj_dir.glob("*.json")
        if f.name != "experiment_summary.json"
    ])
    if not traj_files:
        logger.error(f"No trajectory files found in {traj_dir}")
        return

    if args.task_ids:
        selected_ids = set(args.task_ids.split(','))
        traj_files = [f for f in traj_files if f.stem in selected_ids]

    logger.info(f"Found {len(traj_files)} trajectory files to process")

    # LLM配置
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    model_config = {
        "model_id": os.environ.get("DEFAULT_MODEL", "qwen3-max"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 8000,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
    }

    # 加载所有轨迹数据（需要按顺序，以便提供reference_trajectory）
    all_trajectories = []
    for traj_file in traj_files:
        with open(traj_file, 'r', encoding='utf-8') as f:
            all_trajectories.append((traj_file.stem, json.load(f)))

    # 按task_order排序
    all_trajectories.sort(key=lambda x: x[1].get("task_order", 0))

    # 逐个处理
    results = []
    skipped_count = 0
    llm_called_count = 0

    for i, (task_id, trajectory_data) in enumerate(all_trajectories):
        logger.info(f"\n[{i+1}/{len(all_trajectories)}] Processing: {task_id}")

        # 获取前一个任务的轨迹作为reference
        prev_traj = all_trajectories[i-1][1] if i > 0 else None

        # 构建模板上下文
        context = build_template_context(trajectory_data, prev_traj)

        logger.info(f"  is_success={context['is_success']}, "
                    f"memory_count_before={context['memory_count_before']}, "
                    f"has_reference={context['reference_trajectory'] is not None}")

        # 对于成功的任务，如果启用了skip_llm_on_success，直接返回skipped
        if args.skip_llm_on_success and context['is_success']:
            extracted = {
                "root_cause_conclusion": None,
                "state_mismatch_analysis": None,
                "divergence_point": None,
                "failure_knowledge_graph": [],
                "skipped": True,
                "skipped_reason": "is_success is true; diagnostics is failure-only."
            }
            skipped_count += 1
            logger.info(f"  Skipped (success task, no LLM call)")
        elif args.skip_llm_on_failure and not context['is_success']:
            extracted = {
                "agent_workflow": None,
                "search_workflow": None,
                "skipped": True,
                "skipped_reason": "is_success is false; extraction is success-only."
            }
            skipped_count += 1
            logger.info(f"  Skipped (failed task, no LLM call)")
        else:
            # 渲染Jinja2模板
            try:
                filled_prompt = render_prompt(prompt_template, context)
            except Exception as e:
                logger.error(f"  Template rendering failed: {e}")
                results.append({
                    "task_id": task_id,
                    "task_order": trajectory_data.get("task_order"),
                    "question": trajectory_data.get("question", ""),
                    "is_success": context['is_success'],
                    "error": f"Template rendering failed: {e}",
                    "extracted_memory": None,
                })
                continue

            # 调用LLM
            try:
                response_text = call_llm(filled_prompt, model_config)
                extracted = parse_json_from_response(response_text)
                llm_called_count += 1
                logger.info(f"  Extracted: {json.dumps(extracted, ensure_ascii=False)[:200]}...")
            except Exception as e:
                logger.error(f"  LLM call failed: {e}")
                results.append({
                    "task_id": task_id,
                    "task_order": trajectory_data.get("task_order"),
                    "question": trajectory_data.get("question", ""),
                    "is_success": context['is_success'],
                    "error": str(e),
                    "extracted_memory": None,
                })
                continue

        result = {
            "task_id": task_id,
            "task_order": trajectory_data.get("task_order"),
            "question": trajectory_data.get("question", ""),
            "is_success": context['is_success'],
            "memory_count_before": context['memory_count_before'],
            "has_memory_guidance": bool(context['memory_guidance']),
            "has_reference_trajectory": context['reference_trajectory'] is not None,
            "extracted_memory": extracted,
        }
        results.append(result)

    # 保存结果
    output = {
        "experiment": "memory_extraction_test",
        "prompt_source": args.prompt_file if args.prompt_file else "built-in",
        "total_tasks": len(results),
        "successful_extractions": sum(
            1 for r in results
            if r.get("extracted_memory") is not None
            and not (isinstance(r["extracted_memory"], dict) and r["extracted_memory"].get("_parse_error"))
        ),
        "llm_calls": llm_called_count,
        "skipped_success": skipped_count,
        "timestamp": datetime.now().isoformat(),
        "model": model_config["model_id"],
        "results": results,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total tasks: {len(results)}")
    logger.info(f"LLM calls made: {llm_called_count}")
    logger.info(f"Skipped (success): {skipped_count}")
    success_count = sum(1 for r in results if r.get("is_success"))
    fail_count = len(results) - success_count
    logger.info(f"Success/Fail: {success_count}/{fail_count}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
