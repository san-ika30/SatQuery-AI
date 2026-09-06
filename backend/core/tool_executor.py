"""
SatQuery AI — Tool Executor (Chunk 11)
Orchestrates synchronous and asynchronous execution of registered specialist tools
according to an Agent ToolPlan, preserving execution time, outputs, errors, and evidence.
"""
from __future__ import annotations
import time
import structlog
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

from core.model_registry import get_tool_registry
from core.agent_planner import ToolPlan, PlanStep

log = structlog.get_logger()


@dataclass
class ToolExecutionResult:
    tool: str
    step: int
    status: str  # "success" or "failed"
    execution_time_ms: int
    confidence: float
    output: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    overlay: Optional[str] = None
    error: Optional[str] = None
    fallback_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ToolExecutor:
    """
    Executes tool plans against the central tool registry.
    Propagates context (e.g. detections from grounding to VLM) across steps.
    """

    def __init__(self):
        self.registry = get_tool_registry()

    async def execute_plan(
        self,
        plan: ToolPlan,
        inputs: Dict[str, Any],
        query: str,
    ) -> List[ToolExecutionResult]:
        """
        Execute all steps in an Agent ToolPlan sequentially, passing cumulative evidence.
        """
        results: List[ToolExecutionResult] = []
        cumulative_context = dict(inputs)
        cumulative_context["query"] = query
        cumulative_context["question"] = query

        for plan_step in plan.steps:
            t0 = time.monotonic()
            step_inputs = dict(cumulative_context)
            step_inputs.update(plan_step.parameters)

            log.info("executing_plan_step", step=plan_step.step, tool=plan_step.tool, purpose=plan_step.purpose)

            try:
                exec_dict = await self.registry.execute_tool(plan_step.tool, step_inputs)
                elapsed = exec_dict.get("execution_time_ms", round((time.monotonic() - t0) * 1000))

                res = ToolExecutionResult(
                    tool=plan_step.tool,
                    step=plan_step.step,
                    status=exec_dict.get("status", "success"),
                    execution_time_ms=elapsed,
                    confidence=exec_dict.get("confidence", 0.85),
                    output=exec_dict.get("output", {}),
                    evidence=exec_dict.get("evidence", []),
                    answer=exec_dict.get("answer"),
                    overlay=exec_dict.get("overlay"),
                    error=exec_dict.get("error"),
                    fallback_available=exec_dict.get("fallback_available", False),
                )

                # Feed forward detections if produced
                if "detections" in exec_dict and exec_dict["detections"]:
                    cumulative_context["detections"] = exec_dict["detections"]
                if "overlay" in exec_dict and exec_dict["overlay"]:
                    cumulative_context["last_overlay"] = exec_dict["overlay"]

                results.append(res)

                # If tool failed and no fallback, log warning
                if res.status == "failed":
                    log.warning("tool_step_failed", tool=plan_step.tool, error=res.error)

            except Exception as exc:
                elapsed = round((time.monotonic() - t0) * 1000)
                log.error("tool_execution_exception", tool=plan_step.tool, error=str(exc))
                results.append(ToolExecutionResult(
                    tool=plan_step.tool,
                    step=plan_step.step,
                    status="failed",
                    execution_time_ms=elapsed,
                    confidence=0.0,
                    output={},
                    evidence=[],
                    answer=None,
                    error=str(exc),
                    fallback_available=True,
                ))

        return results


_tool_executor: Optional[ToolExecutor] = None

def get_tool_executor() -> ToolExecutor:
    """Singleton getter for ToolExecutor."""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor()
    return _tool_executor
