"""
Test: Convert GAIA Task 1 extraction results into unified MemoryUnit schema.

Reads extraction outputs from all 5 prompt modules, converts to MemoryUnits,
and saves to task1_memory_units.json for inspection.
"""
import json
import sys
sys.path.insert(0, ".")

from EvolveLab.memory_schema import (
    MemoryUnit, MemoryUnitType, MemoryRelation, RelationType,
    split_extraction_output,
)

TASK1_ID = "04a04a9b-226c-43fd-b319-d5e89743676f"

EXTRACTION_FILES = {
    MemoryUnitType.INSIGHT:    "memory_extraction_insights.json",
    MemoryUnitType.TIP:        "memory_extraction_tips.json",
    MemoryUnitType.TRAJECTORY: "memory_extraction_trajectory.json",
    MemoryUnitType.WORKFLOW:   "memory_extraction_workflow.json",
    MemoryUnitType.SHORTCUT:   "memory_extraction_shortcuts.json",
}


def main():
    all_units = []

    for unit_type, filepath in EXTRACTION_FILES.items():
        with open(filepath) as f:
            data = json.load(f)

        model = data.get("model", "unknown")

        # Find task 1
        task_record = None
        for r in data["results"]:
            if r["task_id"] == TASK1_ID:
                task_record = r
                break

        if task_record is None:
            print(f"  {unit_type.value:12s} | NOT FOUND in {filepath}")
            continue

        mem = task_record.get("extracted_memory", {})

        # Check if skipped
        if isinstance(mem, dict) and mem.get("skipped", False):
            print(f"  {unit_type.value:12s} | SKIPPED ({mem.get('skipped_reason', '')})")
            continue

        # Convert to MemoryUnits
        units = split_extraction_output(
            extraction_result=mem,
            unit_type=unit_type,
            source_task_id=task_record["task_id"],
            source_task_query=task_record["question"][:120],
            task_outcome="success" if task_record["is_success"] else "failure",
            extraction_model=model,
        )

        # Add COOCCURS relations between units from same module
        for i, u in enumerate(units):
            for j, other in enumerate(units):
                if i != j:
                    u.relations.append(MemoryRelation(
                        target_id=other.id,
                        relation_type=RelationType.COOCCURS,
                    ))

        all_units.extend(units)
        token_sum = sum(u.storage_tokens for u in units)
        print(f"  {unit_type.value:12s} | {len(units)} unit(s), {token_sum} tokens")

    # Summary
    ids_by_type = {}
    for u in all_units:
        ids_by_type.setdefault(u.type.value, []).append(u.id)

    print(f"\n  Total: {len(all_units)} MemoryUnits")
    print(f"  By type: { {t: len(ids) for t, ids in ids_by_type.items()} }")
    print(f"  Total storage tokens: {sum(u.storage_tokens for u in all_units)}")

    # RL state vectors preview
    print("\n  RL state vectors (first 3 units):")
    for u in all_units[:3]:
        vec = u.to_rl_state()
        print(f"    {u.type.value:12s} | {vec}")

    # Output
    output = {
        "task_id": TASK1_ID,
        "task_query": all_units[0].source_task_query if all_units else "",
        "task_outcome": "success",
        "total_units": len(all_units),
        "units_by_type": {t: len(ids) for t, ids in ids_by_type.items()},
        "total_storage_tokens": sum(u.storage_tokens for u in all_units),
        "memory_units": [u.to_dict() for u in all_units],
    }

    out_path = "task1_memory_units.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
