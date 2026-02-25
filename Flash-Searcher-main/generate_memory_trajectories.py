#!/usr/bin/env python
# coding=utf-8
"""
Generate Memory Interaction Trajectories

Runs GAIA tasks sequentially with memory accumulating across tasks:
  task1 (no memory) → learn → task2 (uses task1's memory) → learn → task3 (uses task1+2's memory) → ...

Each task is unique. Memory grows as the agent encounters more tasks,
capturing how experience from earlier tasks helps solve later ones.

Output: memory_trajectories/{task_id}.json
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override=True)

from FlashOAgents import OpenAIServerModel
from FlashOAgents.mm_tools import (
    VisualInspectorTool, TextInspectorTool, AudioInspectorTool,
    get_single_file_description, get_zip_description,
)
from base_agent import MMSearchAgent
from utils import read_jsonl
from lasj import judge_equivalence
from EvolveLab.memory_types import MemoryType, TrajectoryData, PROVIDER_MAPPING
from EvolveLab.config import get_memory_config
from eval_utils import TaskTimer, TokenCounter, enrich_result_with_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_memory_provider(memory_type_str, model=None):
    """Load and initialize memory provider from type string"""
    try:
        memory_type = MemoryType(memory_type_str)
    except ValueError:
        logger.error(f"Invalid memory type: {memory_type_str}")
        return None

    if memory_type not in PROVIDER_MAPPING:
        logger.error(f"Memory type {memory_type_str} not found in PROVIDER_MAPPING")
        return None

    try:
        class_name, module_name = PROVIDER_MAPPING[memory_type]
        module = __import__(f"EvolveLab.providers.{module_name}", fromlist=[class_name])
        provider_class = getattr(module, class_name)
        config = get_memory_config(memory_type)
        if model is not None:
            config["model"] = model
        provider = provider_class(config=config)

        if not provider.initialize():
            logger.error(f"Failed to initialize memory provider: {memory_type_str}")
            return None

        logger.info(f"Memory provider loaded: {memory_type_str}")
        return provider
    except Exception as e:
        logger.error(f"Failed to load memory provider {memory_type_str}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_single_task(item, model_config, memory_provider, task_order, max_steps=20, judge_model=None, visual_tool=None, text_tool=None, audio_tool=None):
    """Process a single GAIA task and return trajectory dict"""
    task_model = OpenAIServerModel(**model_config)
    task_model.reset_total_counts()

    search_agent = MMSearchAgent(
        task_model,
        summary_interval=8,
        prompts_type="default",
        max_steps=max_steps,
        memory_provider=memory_provider
    )

    question = item["Question"]
    golden_answer = item["Final answer"]
    task_id = item.get("task_id", f"task-{task_order}")
    level = item.get("Level", "unknown")

    # Handle file attachments
    if item.get("file_name"):
        file_path = f"data/gaia/validation/{item['file_name']}"
        if ".zip" in item["file_name"]:
            question += "\n\nTo solve the task above, you will have to use these attached files:\n"
            question += get_zip_description(
                file_path, question, visual_tool, text_tool, audio_tool,
            )
        else:
            question += "\n\nTo solve the task above, you will have to use this attached file:"
            question += get_single_file_description(
                file_path, question, visual_tool, text_tool, audio_tool,
            )

    timer = TaskTimer()
    timer.start()

    try:
        result = search_agent(question)

        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []

        trajectory = result.get("agent_trajectory", [])

        is_correct = False
        judgement = None
        if judge_model:
            try:
                eval_res = judge_equivalence(
                    question,
                    golden_answer,
                    result.get("agent_result", {}),
                    model=judge_model,
                )
                judgement = eval_res.get("judgement")
                judgement_str = eval_res.get("judgement", "").strip().lower()
                is_correct = (judgement_str == "correct")
            except Exception as e:
                logger.warning(f"Judgement failed: {e}")

        # Take in memory from successful tasks
        if memory_provider:
            try:
                trajectory_data = TrajectoryData(
                    query=question,
                    trajectory=agent_messages,
                    result=result.get("agent_result"),
                    metadata={
                        "task_id": task_id,
                        "status": "success",
                        "is_correct": is_correct,
                        "full_query": question,
                    }
                )
                success, msg = memory_provider.take_in_memory(trajectory_data)
                if success:
                    logger.info(f"Memory ingested for task {task_id}")
                else:
                    logger.warning(f"Memory ingestion skipped for task {task_id}: {msg}")
            except Exception as e:
                logger.warning(f"take_in_memory failed for task {task_id}: {e}")

        token_counter = TokenCounter.from_model(task_model)

        task_result = {
            "agent_result": result.get("agent_result"),
            "judgement": judgement,
            "task_id": task_id,
            "task_order": task_order,
            "level": level,
            "question": question,
            "full_query": question,
            "golden_answer": golden_answer,
            "status": "success",
            "memory_count_before": task_order - 1,
            "agent_trajectory": trajectory,
            "agent_messages": agent_messages,
        }

        timer.stop()
        return enrich_result_with_metrics(task_result, timer, token_counter), is_correct

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Exception processing task {task_id}: {error_msg}")

        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []

        task_result = {
            "agent_result": None,
            "judgement": None,
            "status": "error",
            "error": str(e),
            "task_id": task_id,
            "task_order": task_order,
            "level": level,
            "question": question,
            "full_query": question,
            "golden_answer": golden_answer,
            "memory_count_before": task_order - 1,
            "agent_trajectory": [],
            "agent_messages": agent_messages,
        }

        timer.stop()
        token_counter = TokenCounter.from_model(task_model)
        return enrich_result_with_metrics(task_result, timer, token_counter), False


def main():
    parser = argparse.ArgumentParser(description='Generate Memory Interaction Trajectories')
    parser.add_argument('--infile', type=str,
                        default="./data/gaia/validation/metadata.jsonl",
                        help='Input GAIA dataset file')
    parser.add_argument('--output_dir', type=str,
                        default="./memory_trajectories",
                        help='Output directory for trajectories')
    parser.add_argument('--task_indices', type=str, default=None,
                        help='Task indices to run (e.g., "1,2,3,4,5")')
    parser.add_argument('--max_tasks', type=int, default=None,
                        help='Max number of tasks to run')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Max agent steps per task')
    parser.add_argument('--memory_provider', type=str, default="agent_kb",
                        help='Memory provider to use')
    parser.add_argument('--clear_memory', action='store_true', default=True,
                        help='Clear memory storage before starting')
    parser.add_argument('--no_clear_memory', dest='clear_memory', action='store_false',
                        help='Keep existing memory storage')
    parser.add_argument('--use_dmx', action='store_true', default=False,
                        help='Use DMX/OpenAI API instead of Qwen')
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

    if args.use_dmx:
        model_config = {
            "model_id": os.environ.get("DMX_MODEL", "gpt-4o"),
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": 16000,
            "api_key": os.environ.get("DMX_API_KEY"),
            "api_base": os.environ.get("DMX_API_BASE"),
        }
    else:
        model_config = {
            "model_id": os.environ.get("DEFAULT_MODEL", "qwen-plus"),
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": 8000,
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_base": os.environ.get("OPENAI_API_BASE"),
        }

    # Qwen config for auxiliary tasks (judge, memory, file inspection)
    qwen_config = {
        "model_id": os.environ.get("DEFAULT_MODEL", "qwen-plus"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 8000,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
    }

    # Judge always uses Qwen (lasj.py client is hardcoded to OPENAI_API_KEY/BASE env vars)
    judge_model = os.environ.get("DEFAULT_JUDGE_MODEL", os.environ.get("DEFAULT_MODEL", "qwen-plus"))

    # Load tasks
    data = read_jsonl(args.infile)
    for idx, item in enumerate(data):
        item["_global_index"] = idx + 1

    if args.task_indices:
        indices = [int(x.strip()) for x in args.task_indices.split(',')]
        data = [data[i-1] for i in indices if 0 < i <= len(data)]

    if args.max_tasks and len(data) > args.max_tasks:
        data = data[:args.max_tasks]

    logger.info(f"Loaded {len(data)} tasks from {args.infile}")
    logger.info(f"Memory accumulates across tasks sequentially: task1 → task2 → ... → task{len(data)}")

    # Clear memory storage if requested
    if args.clear_memory:
        storage_path = f"./storage/{args.memory_provider}"
        if os.path.exists(storage_path):
            shutil.rmtree(storage_path)
            logger.info(f"Cleared memory storage: {storage_path}")

    # Initialize models
    # - Agent model (DMX/gpt-4o when --use_dmx, otherwise Qwen) for task execution
    # - Qwen model for auxiliary tasks: memory provider, file inspection
    agent_model = OpenAIServerModel(**model_config)
    qwen_model = OpenAIServerModel(**qwen_config)

    logger.info(f"Agent model: {model_config['model_id']} | Auxiliary model: {qwen_config['model_id']}")

    memory_provider = load_memory_provider(args.memory_provider, qwen_model)
    if not memory_provider:
        logger.error("Failed to load memory provider, exiting")
        return

    # Create inspection tools for file attachments (use Qwen to save cost)
    visual_tool = VisualInspectorTool(qwen_model, 100000)
    text_tool = TextInspectorTool(qwen_model, 100000)
    audio_tool = AudioInspectorTool(qwen_model, 100000)

    # Run tasks sequentially with memory accumulation
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting sequential task execution with memory accumulation")
    logger.info(f"Total tasks: {len(data)} | Memory provider: {args.memory_provider}")
    logger.info(f"{'='*60}\n")

    results = []
    correct_count = 0

    for task_order, task in enumerate(data, 1):
        task_id = task.get("task_id", f"task-{task_order}")
        logger.info(f"\n{'─'*60}")
        logger.info(f"Task {task_order}/{len(data)}: {task_id}")
        logger.info(f"  Question: {task['Question'][:100]}...")
        logger.info(f"  Memory entries available: {task_order - 1}")
        logger.info(f"{'─'*60}")

        result, is_correct = process_single_task(
            task, model_config, memory_provider, task_order,
            max_steps=args.max_steps, judge_model=judge_model,
            visual_tool=visual_tool, text_tool=text_tool, audio_tool=audio_tool
        )

        if is_correct:
            correct_count += 1

        # Save individual task trajectory
        task_file = output_dir / f"{task_id}.json"
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        results.append(result)
        logger.info(f"  Result: {'CORRECT' if is_correct else 'INCORRECT'} "
                    f"| Answer: {str(result.get('agent_result', 'N/A'))[:100]}")
        logger.info(f"  Progress: {correct_count}/{task_order} correct so far "
                    f"({correct_count/task_order*100:.1f}%)")

    # Save experiment summary
    summary = {
        "experiment": "memory_interaction_trajectories",
        "description": "Sequential tasks with cross-task memory accumulation",
        "memory_provider": args.memory_provider,
        "total_tasks": len(data),
        "correct_tasks": correct_count,
        "accuracy": correct_count / len(data) if data else 0,
        "timestamp": datetime.now().isoformat(),
        "model": model_config["model_id"],
        "task_results": [
            {
                "task_order": r.get("task_order"),
                "task_id": r.get("task_id"),
                "question": r.get("question", "")[:100],
                "is_correct": r.get("judgement", "").strip().lower() == "correct" if r.get("judgement") else False,
                "agent_result": str(r.get("agent_result", ""))[:200],
                "golden_answer": r.get("golden_answer", ""),
                "memory_count_before": r.get("memory_count_before", 0),
                "has_memory_guidance": any(
                    step.get("memory_guidance") is not None
                    for step in r.get("agent_trajectory", [])
                ),
            }
            for r in results
        ]
    }

    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results: {correct_count}/{len(data)} correct ({correct_count/len(data)*100:.1f}%)")
    logger.info(f"Trajectories saved to: {output_dir}")
    logger.info(f"Memory accumulated across {len(data)} unique tasks")


if __name__ == '__main__':
    main()
