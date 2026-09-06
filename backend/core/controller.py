"""
SatQuery AI — Agentic Controller
Uses Mistral-7B (HF Inference API) to classify the task, with a fast
rule-based fallback when the API is unavailable or slow.
"""
from __future__ import annotations
import json
import os
import re
import time
import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from schemas.models import InputMode, TaskPlan, TaskType

log = structlog.get_logger()

_HF_TOKEN_RAW = os.getenv("HF_API_TOKEN", "")
# Treat placeholder / unset tokens as empty so rule-based fallback activates
HF_API_TOKEN = _HF_TOKEN_RAW if _HF_TOKEN_RAW.startswith("hf_") and len(_HF_TOKEN_RAW) > 10 else ""
HF_MISTRAL_MODEL = os.getenv("HF_MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{HF_MISTRAL_MODEL}"

SYSTEM_PROMPT = """You are SatQuery AI's task classifier for remote sensing analysis.
Given a user query and input mode, return ONLY a valid JSON object with this schema:
{
  "task_type": one of ["vqa", "caption", "grounding", "change_detection", "change_vqa", "sar_fusion"],
  "models": list of model names from ["geochat", "grounding_dino", "changeformer", "sar_fusion_encoder"],
  "parameters": {},
  "reasoning": "brief explanation"
}

Rules:
- "vqa": user asks a question about a single image content
- "caption": user asks to describe/caption/summarize a single image
- "grounding": user asks to find/highlight/locate a specific region or object
- "change_detection": user asks what changed, changed area, change map (bi-temporal input)
- "change_vqa": user asks a question about change between two dates (bi-temporal input)
- "sar_fusion": user asks about cross-modal optical+SAR joint analysis
- If input_mode is "bitemporal", prefer change_detection or change_vqa
- If input_mode is "crossmodal", prefer sar_fusion
- Always include geochat for vqa/caption/change_vqa tasks
- For single-image vqa questions about objects or spatial content (water, buildings, roads, objects), include both grounding_dino and geochat
- Always include grounding_dino for grounding tasks
- Always include changeformer for change_detection/change_vqa tasks
- Always include sar_fusion_encoder for sar_fusion tasks"""


# ── Rule-based fallback classifier ───────────────────────────────────────────

_GROUNDING_KW = re.compile(
    r"\b(find|locate|highlight|show|where is|mark|identify region|point out|bbox|bounding box)\b",
    re.IGNORECASE,
)
_CAPTION_KW = re.compile(
    r"\b(describe|caption|summarize|what is in|what do you see|explain|tell me about)\b",
    re.IGNORECASE,
)
_CHANGE_KW = re.compile(
    r"\b(change|changed|difference|before|after|temporal|increase|decrease|grew|shrunk)\b",
    re.IGNORECASE,
)
_SAR_KW = re.compile(
    r"\b(sar|radar|backscatter|optical|fusion|combine|cross.modal|complementary)\b",
    re.IGNORECASE,
)
_SPATIAL_VQA_KW = re.compile(
    r"\b(water|water body|water bodies|lake|river|ocean|building|buildings|structure|structures|house|houses|road|roads|highway|car|cars|vehicle|vehicles|tree|trees|forest|vegetation|object|objects|count|how many|is there|are there|any|visible|present|detect)\b",
    re.IGNORECASE,
)


def _rule_based_classify(query: str, input_mode: InputMode) -> TaskPlan:
    """Fast keyword-based task classifier — fallback when Mistral is unavailable."""
    query_lower = query.lower()

    if input_mode == InputMode.crossmodal:
        return TaskPlan(
            task_type=TaskType.sar_fusion,
            input_mode=input_mode,
            models=["sar_fusion_encoder", "geochat"],
            parameters={},
            reasoning="Cross-modal input detected → SAR fusion task (rule-based).",
        )

    if input_mode == InputMode.bitemporal:
        # If question mark, it's change_vqa; otherwise change_detection
        if "?" in query or any(w in query_lower for w in ["has", "did", "is", "are", "how much"]):
            return TaskPlan(
                task_type=TaskType.change_vqa,
                input_mode=input_mode,
                models=["changeformer", "geochat"],
                parameters={},
                reasoning="Bi-temporal input with question → change VQA (rule-based).",
            )
        return TaskPlan(
            task_type=TaskType.change_detection,
            input_mode=input_mode,
            models=["changeformer"],
            parameters={},
            reasoning="Bi-temporal input → change detection (rule-based).",
        )

    # Single image
    if _GROUNDING_KW.search(query):
        return TaskPlan(
            task_type=TaskType.grounding,
            input_mode=input_mode,
            models=["grounding_dino"],
            parameters={},
            reasoning="Grounding keywords detected (rule-based).",
        )
    if _SPATIAL_VQA_KW.search(query):
        return TaskPlan(
            task_type=TaskType.vqa,
            input_mode=input_mode,
            models=["grounding_dino", "geochat"],
            parameters={},
            reasoning="Single-image VQA with spatial object localization (rule-based).",
        )
    if _CAPTION_KW.search(query):
        return TaskPlan(
            task_type=TaskType.caption,
            input_mode=input_mode,
            models=["geochat"],
            parameters={},
            reasoning="Caption/describe keywords detected (rule-based).",
        )
    # Default: VQA
    return TaskPlan(
        task_type=TaskType.vqa,
        input_mode=input_mode,
        models=["geochat"],
        parameters={},
        reasoning="Default to VQA for single image question (rule-based).",
    )


# ── Mistral-7B via HF Inference API ──────────────────────────────────────────

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_mistral(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.1,
            "return_full_text": False,
        },
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(HF_INFERENCE_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        return ""


def _parse_mistral_response(text: str, input_mode: InputMode) -> TaskPlan | None:
    """Extract JSON from Mistral output and parse into TaskPlan."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        return TaskPlan(
            task_type=TaskType(obj["task_type"]),
            input_mode=input_mode,
            models=obj.get("models", ["geochat"]),
            parameters=obj.get("parameters", {}),
            reasoning=obj.get("reasoning", ""),
        )
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        log.warning("mistral_parse_failed", error=str(e))
        return None


# ── Public API ────────────────────────────────────────────────────────────────

async def classify_task(query: str, input_mode: InputMode) -> tuple[TaskPlan, str]:
    """
    Classify the user query into a TaskPlan.

    Returns:
        (TaskPlan, method)  where method is "mistral" or "rule_based"
    """
    if not HF_API_TOKEN:
        log.info("no_hf_token_using_rule_based_classifier")
        return _rule_based_classify(query, input_mode), "rule_based"

    prompt = (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n"
        f"Input mode: {input_mode.value}\n"
        f"User query: {query}\n\n"
        f"Return only JSON. [/INST]"
    )

    try:
        t0 = time.monotonic()
        mistral_text = await _call_mistral(prompt)
        elapsed = round((time.monotonic() - t0) * 1000)
        log.info("mistral_response", elapsed_ms=elapsed)

        plan = _parse_mistral_response(mistral_text, input_mode)
        if plan:
            log.info("task_classified_by_mistral", task=plan.task_type.value)
            return plan, "mistral"
        log.warning("mistral_output_unparseable_falling_back")
    except Exception as exc:
        log.warning("mistral_api_failed", error=str(exc))

    # Fallback
    plan = _rule_based_classify(query, input_mode)
    log.info("task_classified_by_rules", task=plan.task_type.value)
    return plan, "rule_based"
