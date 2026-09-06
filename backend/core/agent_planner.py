"""
SatQuery AI — Agent Planner (Chunk 11)
Converts structured QueryIntent and available input payloads into an executable multi-step ToolPlan.
"""
from __future__ import annotations
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

from core.query_interpreter import QueryIntent
from core.model_registry import get_tool_registry


@dataclass
class PlanStep:
    step: int
    tool: str
    purpose: str
    inputs_required: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolPlan:
    plan_id: str
    intent: str
    sub_intent: str
    steps: List[PlanStep]
    combiner_required: bool
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "intent": self.intent,
            "sub_intent": self.sub_intent,
            "steps": [s.to_dict() for s in self.steps],
            "combiner_required": self.combiner_required,
            "reasoning": self.reasoning,
        }


class AgentPlanner:
    """
    Dynamic planner constructing ordered execution graphs of registered specialist tools.
    """

    def __init__(self):
        self.registry = get_tool_registry()

    def create_plan(
        self,
        query: str,
        intent: QueryIntent,
        inputs: Dict[str, Any],
    ) -> ToolPlan:
        """
        Synthesize an executable ToolPlan matching the user query intent and input availability.
        """
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        steps: List[PlanStep] = []
        combiner_required = False
        reasoning = ""

        q_lower = query.lower()

        # ── 1. Complex Multi-Tool Query ───────────────────────────────────────
        if intent.intent == "complex_multitool":
            combiner_required = True
            step_idx = 1

            if intent.requires_change_detection:
                steps.append(PlanStep(
                    step=step_idx,
                    tool="change_analyzer",
                    purpose="Analyze bi-temporal change between pre- and post-change dates",
                    inputs_required=["image_t1", "image_t2"],
                    parameters={"query": query},
                ))
                step_idx += 1

            if intent.requires_cross_modal:
                steps.append(PlanStep(
                    step=step_idx,
                    tool="optical_sar_analyzer",
                    purpose="Perform cross-modal optical + SAR feature fusion and spatial cross-validation",
                    inputs_required=["image_optical", "image_sar"],
                    parameters={"query": query},
                ))
                step_idx += 1

            if intent.requires_grounding:
                target_prompt = " ".join(intent.objects) if intent.objects else "building"
                steps.append(PlanStep(
                    step=step_idx,
                    tool="grounding_specialist",
                    purpose=f"Locate and segment candidate objects ({target_prompt})",
                    inputs_required=["image"],
                    parameters={"prompt": target_prompt},
                ))
                step_idx += 1

            # Fallback if no specific step was queued
            if not steps:
                steps.append(PlanStep(
                    step=1,
                    tool="satellite_vlm",
                    purpose="Reason over scene with Satellite VLM",
                    inputs_required=["image", "question"],
                    parameters={"question": query},
                ))

            reasoning = "Multi-stage compound query planned: coordinating change detection, cross-modal verification, and visual grounding."

        # ── 2. Optical + SAR Cross-Modal Query ─────────────────────────────────
        elif intent.intent == "optical_sar_crossmodal":
            steps.append(PlanStep(
                step=1,
                tool="optical_sar_analyzer",
                purpose="Execute dual-stream optical-SAR tensor fusion and mutual spatial cross-validation",
                inputs_required=["image_optical", "image_sar"],
                parameters={"query": query},
            ))

            if intent.requires_grounding:
                combiner_required = True
                target_prompt = " ".join(intent.objects) if intent.objects else "water body. building."
                steps.append(PlanStep(
                    step=2,
                    tool="grounding_specialist",
                    purpose=f"Localize target objects ({target_prompt}) in optical imagery",
                    inputs_required=["image"],
                    parameters={"prompt": target_prompt},
                ))
                reasoning = "Optical + SAR cross-modal fusion planned with secondary visual grounding for spatial bounding boxes."
            else:
                reasoning = "Optical + SAR dual-stream neural fusion and spatial cross-validation planned."

        # ── 3. Bi-Temporal Change Detection & VQA ─────────────────────────────
        elif intent.intent == "bitemporal_change":
            steps.append(PlanStep(
                step=1,
                tool="change_analyzer",
                purpose="Execute ChangeFormer transformer inference for bi-temporal change detection and Change-VQA",
                inputs_required=["image_t1", "image_t2"],
                parameters={"query": query},
            ))

            # If question asks fine-grained questions, invoke VLM as secondary step
            if intent.sub_intent == "change_vqa" and ("?" in query or any(w in q_lower for w in ["what", "how", "did", "why"])):
                combiner_required = True
                steps.append(PlanStep(
                    step=2,
                    tool="satellite_vlm",
                    purpose="Formulate natural-language reasoning incorporating bi-temporal change evidence",
                    inputs_required=["image", "question"],
                    parameters={"question": query},
                ))
                reasoning = "ChangeFormer bi-temporal change detection planned with VLM reasoning for detailed question answering."
            else:
                reasoning = "ChangeFormer bi-temporal change detection and change ratio quantification planned."

        # ── 4. Object Grounding / Spatial Localization ────────────────────────
        elif intent.intent == "object_grounding":
            target_prompt = " ".join(intent.objects) if intent.objects else "satellite object"
            steps.append(PlanStep(
                step=1,
                tool="grounding_specialist",
                purpose=f"Locate, highlight, and segment target regions ({target_prompt}) using Grounding DINO + SAM",
                inputs_required=["image"],
                parameters={"prompt": target_prompt},
            ))

            # If user also asks a question along with grounding (e.g. "Highlight the water body and tell me whether it is present")
            if "?" in query or any(w in q_lower for w in ["tell me", "is it", "whether", "is present", "are present", "how many"]):
                combiner_required = True
                steps.append(PlanStep(
                    step=2,
                    tool="satellite_vlm",
                    purpose="Answer query question using visual and grounding context",
                    inputs_required=["image", "question"],
                    parameters={"question": query},
                ))
                reasoning = f"Compound grounding and question answering planned: Grounding DINO + SAM for '{target_prompt}' followed by Satellite VLM."
            else:
                reasoning = f"Object grounding planned using Grounding DINO + SAM for '{target_prompt}'."

        # ── 5. Optical Spectral Query ─────────────────────────────────────────
        elif intent.intent == "optical_specific":
            steps.append(PlanStep(
                step=1,
                tool="optical_analyzer",
                purpose="Extract optical spectral indices (NDVI vegetation, NDWI water, texture)",
                inputs_required=["image"],
                parameters={"query": query},
            ))
            reasoning = "Optical multispectral analyzer planned for NDVI/NDWI and vegetation/water estimation."

        # ── 6. SAR Radar Query ────────────────────────────────────────────────
        elif intent.intent == "sar_specific":
            steps.append(PlanStep(
                step=1,
                tool="sar_analyzer",
                purpose="Analyze Sentinel-1 C-band SAR polarimetric backscatter (VV/VH dB) and surface roughness",
                inputs_required=["image"],
                parameters={"query": query},
            ))
            reasoning = "Sentinel-1 SAR radar analyzer planned for C-band microwave backscatter and structural evaluation."

        # ── 7. Single-Image Scene Description ─────────────────────────────────
        elif intent.intent == "single_image_description":
            steps.append(PlanStep(
                step=1,
                tool="satellite_vlm",
                purpose="Generate comprehensive natural-language scene caption and land-cover description",
                inputs_required=["image", "question"],
                parameters={"question": query},
            ))
            reasoning = "Single-image VLM specialist planned for scene captioning and land-cover description."

        # ── 8. Single-Image VQA (Default) ─────────────────────────────────────
        else:
            if intent.requires_grounding:
                combiner_required = True
                target_prompt = " ".join(intent.objects) if intent.objects else "building. road. water body."
                steps.append(PlanStep(
                    step=1,
                    tool="grounding_specialist",
                    purpose=f"Localize objects of interest ({target_prompt}) to support counting/spatial evidence",
                    inputs_required=["image"],
                    parameters={"prompt": target_prompt},
                ))
                steps.append(PlanStep(
                    step=2,
                    tool="satellite_vlm",
                    purpose="Perform visual question answering conditioned on grounding evidence",
                    inputs_required=["image", "question"],
                    parameters={"question": query},
                ))
                reasoning = "Single-image VQA with grounding evidence planned for object counting/spatial reasoning."
            else:
                steps.append(PlanStep(
                    step=1,
                    tool="satellite_vlm",
                    purpose="Execute visual question answering with Satellite VLM reasoning engine",
                    inputs_required=["image", "question"],
                    parameters={"question": query},
                ))
                reasoning = "Single-image VQA reasoning planned with Satellite VLM."

        return ToolPlan(
            plan_id=plan_id,
            intent=intent.intent,
            sub_intent=intent.sub_intent,
            steps=steps,
            combiner_required=combiner_required,
            reasoning=reasoning,
        )


_agent_planner: Optional[AgentPlanner] = None

def get_agent_planner() -> AgentPlanner:
    """Singleton getter for AgentPlanner."""
    global _agent_planner
    if _agent_planner is None:
        _agent_planner = AgentPlanner()
    return _agent_planner
