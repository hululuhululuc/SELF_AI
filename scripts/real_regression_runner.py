import asyncio
import json
import time

from self_ai.main import run_autonomy_workflow

TASKS = [
    ("quick_answer", "用一句话解释Redis在这个项目中的作用"),
    ("document_writing", "帮我写一段简短的项目说明，介绍这个多智能体编排系统。"),
    ("code_generation", "写一个Python函数，计算列表中数字的平均值，并处理空列表。"),
    ("architecture_design", "为当前项目设计下一阶段公开数据集评测方案。"),
    ("research_summary", "总结这个项目中Redis、Qdrant、Neo4j分别承担什么作用。"),
]


async def main() -> None:
    rows = []
    for name, task in TASKS:
        t0 = time.perf_counter()
        try:
            result = await run_autonomy_workflow(task)
            dt = time.perf_counter() - t0
            rows.append(
                {
                    "case": name,
                    "ok": True,
                    "elapsed_s": round(dt, 2),
                    "run_id": result.get("run_id"),
                    "mode": (result.get("workflow_decision") or {}).get("execution_mode"),
                    "pipeline": (result.get("metadata") or {}).get("pipeline"),
                    "pipeline_source": (result.get("metadata") or {}).get("pipeline_source"),
                    "model": result.get("model"),
                    "selected_agents": result.get("selected_agents"),
                    "revision_count": (result.get("metadata") or {}).get("revision_count"),
                    "quality_gate": (result.get("quality_gate") or {}).get("decision"),
                    "errors": len(result.get("errors") or []),
                    "response_preview": str(result.get("response") or "")[:180],
                }
            )
        except Exception as exc:
            dt = time.perf_counter() - t0
            rows.append(
                {
                    "case": name,
                    "ok": False,
                    "elapsed_s": round(dt, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
