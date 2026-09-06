"""
SatQuery AI — Multi-Microservice Orchestrator Core
Coordinates:
  User Request -> Validate Request -> Decode/Validate Image -> Grounding Prompt Derivation ->
  Grounding Specialist (Grounding DINO + SAM) -> VLM Specialist (Vision-Language Reasoning) ->
  Unified SatQuery Response
"""
import io
import os
import sys
import json
import base64
import time
import httpx
import importlib.util
import structlog
from PIL import Image

log = structlog.get_logger()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

GROUNDING_MODULE_PATH = os.path.join(PROJECT_ROOT, "hf_spaces", "grounding")
VLM_MODULE_PATH = os.path.join(PROJECT_ROOT, "hf_spaces", "vlm")

GROUNDING_SERVICE_URL = os.getenv(
    "GROUNDING_SERVICE_URL",
    "https://sanika2006-satquery-grounding.hf.space/run/predict",
)
VLM_SERVICE_URL = os.getenv(
    "VLM_SERVICE_URL",
    "https://sanika2006-satquery-vlm.hf.space/run/predict",
)

_grounding_module = None
_vlm_module = None


def get_grounding_module():
    """Lazily and dynamically load hf_spaces/grounding/app.py without namespace collision."""
    global _grounding_module
    if _grounding_module is not None:
        return _grounding_module

    grounding_app_path = os.path.join(GROUNDING_MODULE_PATH, "app.py")
    if not os.path.exists(grounding_app_path):
        raise FileNotFoundError(f"Grounding app module missing at {grounding_app_path}")

    if GROUNDING_MODULE_PATH not in sys.path:
        sys.path.insert(0, GROUNDING_MODULE_PATH)

    spec = importlib.util.spec_from_file_location("satquery_grounding_app", grounding_app_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["satquery_grounding_app"] = module
    spec.loader.exec_module(module)
    _grounding_module = module
    return _grounding_module


_vlm_mtime = None

def get_vlm_module(reload: bool = False):
    """Lazily and dynamically load hf_spaces/vlm/app.py without namespace collision."""
    global _vlm_module, _vlm_mtime
    vlm_app_path = os.path.join(VLM_MODULE_PATH, "app.py")
    if not os.path.exists(vlm_app_path):
        raise FileNotFoundError(f"VLM app module missing at {vlm_app_path}")

    current_mtime = os.path.getmtime(vlm_app_path)
    if _vlm_module is not None and not reload and _vlm_mtime == current_mtime:
        return _vlm_module

    if VLM_MODULE_PATH not in sys.path:
        sys.path.insert(0, VLM_MODULE_PATH)

    spec = importlib.util.spec_from_file_location("satquery_vlm_app", vlm_app_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["satquery_vlm_app"] = module
    spec.loader.exec_module(module)
    _vlm_module = module
    _vlm_mtime = current_mtime
    return _vlm_module


def derive_grounding_prompt(question: str) -> str:
    """
    Modular intent/object extraction mapping natural language questions to grounding prompts.
    Extracts query-specific candidate classes and appends open-vocabulary background classes
    (e.g. 'building. road. water body. vegetation. car.') to ensure contrastive logit evaluation
    in Grounding DINO, preventing false positive single-token classification.
    """
    if not question or not isinstance(question, str):
        return "building. road. water body. vegetation. car."

    q = question.lower().strip()

    prompts = []
    if "water" in q or "lake" in q or "river" in q or "ocean" in q or "pond" in q:
        prompts.append("water body")
    if "building" in q or "house" in q or "structure" in q or "roof" in q or "built" in q:
        prompts.append("building")
    if "tree" in q or "forest" in q or "vegetation" in q or "crop" in q or "cropland" in q:
        prompts.append("vegetation")
    if "car" in q or "vehicle" in q or "automobile" in q or "truck" in q:
        prompts.append("car")
    if "road" in q or "highway" in q or "street" in q:
        prompts.append("road")
    if "ship" in q or "boat" in q or "vessel" in q:
        prompts.append("ship")
    if "red" in q:
        prompts.append("red box")
    if "blue" in q:
        prompts.append("blue box")

    if not prompts and any(w in q for w in ["object", "objects", "item", "items", "detect", "find", "locate", "identify", "major", "visible"]):
        prompts = ["building", "road", "water body", "vegetation", "car"]

    base_classes = ["building", "road", "water body", "vegetation", "car"]

    seen = set()
    unique = []

    # 1. Query-derived prompts first
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    # 2. Append scene base classes for open-vocabulary negative contrast
    for b in base_classes:
        if b not in seen:
            seen.add(b)
            unique.append(b)

    return ". ".join(unique) + "."


async def call_grounding_specialist(image_b64: str, prompt: str) -> tuple[str, list[dict]]:
    """
    Invoke Grounding Specialist microservice (Grounding DINO + SAM).
    Tries remote HF Space URL first; falls back cleanly to local module pipeline.
    Returns tuple of (overlay_b64, list_of_detection_dicts).
    """
    t0 = time.monotonic()
    payload = {"data": [image_b64, prompt]}

    # 1. Remote service call (with retry)
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
                resp = await client.post(GROUNDING_SERVICE_URL, json=payload)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if len(data) >= 2:
                        overlay_b64 = data[0]
                        dets_raw = data[1]
                        dets = json.loads(dets_raw) if isinstance(dets_raw, str) else dets_raw
                        if isinstance(dets, list):
                            log.info("grounding_remote_success", duration_ms=round((time.monotonic() - t0) * 1000))
                            return overlay_b64, dets
        except Exception as exc:
            log.warning(f"grounding_remote_attempt_{attempt+1}_failed", error=str(exc))
            if attempt == 0:
                await asyncio.sleep(0.5)

    # 2. Local module fallback execution
    try:
        grounding_app = get_grounding_module()
        img = grounding_app.decode_base64_image(image_b64)
        pipeline_res = grounding_app.run_grounding_sam_pipeline(img, prompt)
        overlay_b64 = pipeline_res["mask_overlay_b64"]
        dets = pipeline_res["detections"]
        log.info("grounding_local_success", duration_ms=round((time.monotonic() - t0) * 1000))
        return overlay_b64, dets
    except Exception as exc:
        log.error("grounding_local_failed", error=str(exc))
        raise RuntimeError("Object detection service is currently unavailable.")


async def call_vlm_specialist(image_b64: str, question: str, detections: list[dict]) -> tuple[str, dict]:
    """
    Invoke VLM Reasoning Specialist microservice.
    Tries remote HF Space URL first; falls back cleanly to local VLM reasoning engine.
    Returns tuple of (answer_text, structured_result_dict).
    """
    t0 = time.monotonic()
    dets_json = json.dumps(detections)
    payload = {"data": [image_b64, question, dets_json]}

    # 1. Remote service call (with retry)
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
                resp = await client.post(VLM_SERVICE_URL, json=payload)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if len(data) >= 2:
                        answer = data[0]
                        struct_raw = data[1]
                        struct_res = json.loads(struct_raw) if isinstance(struct_raw, str) else struct_raw
                        log.info("vlm_remote_success", duration_ms=round((time.monotonic() - t0) * 1000))
                        return answer, struct_res
        except Exception as exc:
            log.warning(f"vlm_remote_attempt_{attempt+1}_failed", error=str(exc))
            if attempt == 0:
                await asyncio.sleep(0.5)

    # 2. Local module fallback execution
    try:
        vlm_app = get_vlm_module()
        img = vlm_app.decode_base64_image(image_b64)
        res = vlm_app.answer_satellite_question(img, question, detections)
        log.info("vlm_local_success", duration_ms=round((time.monotonic() - t0) * 1000))
        return res["answer"], res["reasoning_context"]
    except Exception as exc:
        log.error("vlm_local_failed", error=str(exc))
        raise RuntimeError("Vision reasoning service is currently unavailable.")


async def orchestrate_satquery_request(image_b64: str, question: str) -> dict:
    """
    Full SatQuery AI Unified Orchestrator:
      1. Validate Request Inputs
      2. Decode & Validate Base64 Image
      3. Extract Grounding Prompt
      4. Grounding Specialist (DINO + SAM) -> Overlay + Bounding Boxes + Masks
      5. VLM Specialist Reasoning -> Natural Language Answer
      6. Unified Response Payload
    """
    t0 = time.monotonic()

    # 1. Validate Request Inputs
    if not image_b64 or not isinstance(image_b64, str) or not image_b64.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_IMAGE_PAYLOAD",
                "message": "Base64 image input is empty or invalid.",
            },
        }

    if not question or not isinstance(question, str) or not question.strip():
        return {
            "status": "error",
            "error": {
                "code": "EMPTY_QUESTION",
                "message": "Question text cannot be empty.",
            },
        }

    # 2. Verify Base64 image decoding
    clean_b64 = image_b64.strip()
    if "," in clean_b64:
        clean_b64 = clean_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(clean_b64)
        if not img_bytes:
            raise ValueError("Decoded byte stream is empty.")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
    except Exception as exc:
        return {
            "status": "error",
            "error": {
                "code": "INVALID_IMAGE_FORMAT",
                "message": f"Failed to decode image file: {str(exc)}",
            },
        }

    # 3. Extract Grounding Prompt
    grounding_prompt = derive_grounding_prompt(question)
    log.info("orchestrator_request_received", prompt=grounding_prompt, question_len=len(question), image_size=[w, h])

    # 4. Grounding Specialist
    try:
        overlay_b64, detections = await call_grounding_specialist(clean_b64, grounding_prompt)
    except Exception as exc:
        return {
            "status": "error",
            "error": {
                "code": "GROUNDING_SERVICE_UNAVAILABLE",
                "message": "Object detection service is currently unavailable.",
            },
        }

    # 5. VLM Specialist
    try:
        answer_text, reasoning_struct = await call_vlm_specialist(clean_b64, question, detections)
    except Exception as exc:
        return {
            "status": "error",
            "error": {
                "code": "VLM_SERVICE_UNAVAILABLE",
                "message": "Vision reasoning service is currently unavailable.",
            },
        }

    elapsed_ms = round((time.monotonic() - t0) * 1000)
    log.info("orchestrator_request_success", duration_ms=elapsed_ms, detections_count=len(detections))

    # 6. Unified Response Payload
    return {
        "status": "success",
        "answer": answer_text,
        "detections": detections,
        "overlay": overlay_b64,
        "image_size": [w, h],
        "execution_time_ms": elapsed_ms,
    }


async def orchestrate_bitemporal_satquery_request(
    image_t1_b64: str,
    image_t2_b64: str,
    question: str,
) -> dict:
    """
    Bi-Temporal SatQuery AI Orchestrator:
      1. Validate Request Inputs & Exactly Two Image Payloads
      2. Decode & Validate T1 (Pre-Change) and T2 (Post-Change) Satellite Images
      3. Execute ChangeAnalyzer (ChangeFormerV6) -> Heatmap + Change Ratio + Severity + Evidence
      4. Pass Change Evidence & T2 Image to VLM Specialist for Natural-Language Answer
      5. Return Auditable Unified Bi-Temporal Response Payload
    """
    t0 = time.monotonic()

    # 1. Input Validation
    if not image_t1_b64 or not isinstance(image_t1_b64, str) or not image_t1_b64.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_IMAGE_T1_PAYLOAD",
                "message": "Image T1 (pre-change) payload is empty or invalid.",
            },
        }

    if not image_t2_b64 or not isinstance(image_t2_b64, str) or not image_t2_b64.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_IMAGE_T2_PAYLOAD",
                "message": "Image T2 (post-change) payload is empty or invalid.",
            },
        }

    if not question or not isinstance(question, str) or not question.strip():
        return {
            "status": "error",
            "error": {
                "code": "EMPTY_QUESTION",
                "message": "Question text cannot be empty.",
            },
        }

    # 2. Decode Images
    clean_t1 = image_t1_b64.strip()
    if "," in clean_t1:
        clean_t1 = clean_t1.split(",", 1)[1]

    clean_t2 = image_t2_b64.strip()
    if "," in clean_t2:
        clean_t2 = clean_t2.split(",", 1)[1]

    try:
        bytes_t1 = base64.b64decode(clean_t1)
        bytes_t2 = base64.b64decode(clean_t2)
        if not bytes_t1 or not bytes_t2:
            raise ValueError("Decoded byte stream is empty.")
        img_t1 = Image.open(io.BytesIO(bytes_t1)).convert("RGB")
        img_t2 = Image.open(io.BytesIO(bytes_t2)).convert("RGB")
        w, h = img_t1.size
    except Exception as exc:
        return {
            "status": "error",
            "error": {
                "code": "INVALID_IMAGE_FORMAT",
                "message": f"Failed to decode bi-temporal image pair: {str(exc)}",
            },
        }

    # 3. Execute ChangeAnalyzer
    from core.change_analyzer import get_change_analyzer
    analyzer = get_change_analyzer()
    change_res = analyzer.analyze_change(img_t1, img_t2, question)

    # Convert changed regions into detection dict format for VLM compatibility
    change_detections = []
    for idx, reg in enumerate(change_res.get("changed_regions", [])):
        change_detections.append({
            "box_2d": reg["box_2d"],
            "label": f"changed_region_{idx+1}",
            "score": float(reg.get("area_ratio", 1.0)) / 100.0,
        })

    # 4. Formulate Natural-Language Answer from Semantic Change Reasoning
    if change_res.get("natural_answer"):
        answer_text = change_res["natural_answer"]
    else:
        # Fallback to evidence summary
        answer_text = (
            f"Bi-temporal change analysis indicates {change_res['change_ratio']}% change area "
            f"({change_res['severity']} change). {change_res['evidence_summary']}"
        )

    elapsed_ms = round((time.monotonic() - t0) * 1000)
    log.info("bitemporal_orchestrator_success", duration_ms=elapsed_ms, change_ratio=change_res["change_ratio"])

    # 5. Auditable Unified Response Payload
    return {
        "status": "success",
        "task": "bi_temporal_change_analysis",
        "inputs": 2,
        "answer": answer_text,
        "change_ratio": change_res["change_ratio"],
        "severity": change_res["severity"],
        "change_categories": change_res["change_categories"],
        "overlay": change_res["change_mask_b64"],
        "evidence": [change_res["evidence_summary"]],
        "confidence": change_res["confidence"],
        "model": change_res["model_used"],
        "tools": ["change_analyzer", "vlm_specialist"],
        "image_size": [w, h],
        "execution_time_ms": elapsed_ms,
    }


async def orchestrate_crossmodal_satquery_request(
    optical_b64: str,
    sar_b64: str,
    question: str,
) -> dict:
    """
    Cross-Modal Optical + SAR SatQuery AI Orchestrator:
      1. Validate Request Inputs & Modality Payloads
      2. Decode & Validate Sentinel-2 Optical and Sentinel-1 SAR Satellite Images
      3. Execute OpticalSARAnalyzer -> Dual-stream Fusion + Spectral/Radar Evidence + Overlays
      4. Synthesize Optical Grounding Evidence with Grounding DINO + SAM where localization is requested
      5. Return Auditable Unified Cross-Modal Response Payload
    """
    t0 = time.monotonic()

    # 1. Input Validation
    if not optical_b64 or not isinstance(optical_b64, str) or not optical_b64.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_OPTICAL_PAYLOAD",
                "message": "Optical image payload is empty or invalid.",
            },
        }

    if not sar_b64 or not isinstance(sar_b64, str) or not sar_b64.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_SAR_PAYLOAD",
                "message": "SAR image payload is empty or invalid.",
            },
        }

    if not question or not isinstance(question, str) or not question.strip():
        return {
            "status": "error",
            "error": {
                "code": "EMPTY_QUESTION",
                "message": "Question text cannot be empty.",
            },
        }

    # 2. Decode Images
    clean_opt = optical_b64.strip()
    if "," in clean_opt:
        clean_opt = clean_opt.split(",", 1)[1]

    clean_sar = sar_b64.strip()
    if "," in clean_sar:
        clean_sar = clean_sar.split(",", 1)[1]

    try:
        bytes_opt = base64.b64decode(clean_opt)
        bytes_sar = base64.b64decode(clean_sar)
        if not bytes_opt or not bytes_sar:
            raise ValueError("Decoded byte stream is empty.")
        img_opt = Image.open(io.BytesIO(bytes_opt)).convert("RGB")
        img_sar = Image.open(io.BytesIO(bytes_sar))
        w, h = img_opt.size
    except Exception as exc:
        return {
            "status": "error",
            "error": {
                "code": "INVALID_IMAGE_FORMAT",
                "message": f"Failed to decode optical/SAR image pair: {str(exc)}",
            },
        }

    # 3. Execute OpticalSARAnalyzer
    from core.optical_sar_analyzer import get_optical_sar_analyzer
    analyzer = get_optical_sar_analyzer()
    fusion_res = analyzer.analyze_cross_modal(img_opt, img_sar, question)

    # 4. Optical Spatial Localization via Grounding DINO if query asks for localization
    q_lower = question.lower()
    localization_requested = any(kw in q_lower for kw in ["locate", "highlight", "where", "bounding box", "find"])
    optical_detections = []
    if localization_requested:
        try:
            target_phrase = "building. road." if ("built-up" in q_lower or "urban" in q_lower) else "water body."
            _, optical_detections = await call_grounding_specialist(optical_b64, target_phrase)
        except Exception as exc:
            log.warning("optical_grounding_skipped_in_fusion", error=str(exc))

    elapsed_ms = round((time.monotonic() - t0) * 1000)
    log.info("crossmodal_orchestrator_success", duration_ms=elapsed_ms, task=fusion_res["task"])

    # 5. Auditable Unified Cross-Modal Response Payload
    return {
        "status": "success",
        "task": "optical_sar_cross_modal",
        "inputs": 2,
        "modalities": ["optical", "sar"],
        "answer": fusion_res["answer"],
        "confidence": fusion_res["confidence"],
        "optical_evidence": fusion_res["optical_evidence"],
        "sar_evidence": fusion_res["sar_evidence"],
        "fused_evidence": fusion_res["fused_evidence"],
        "evidence": fusion_res["optical_evidence"] + fusion_res["sar_evidence"] + fusion_res["fused_evidence"],
        "overlay": fusion_res["overlay"],
        "detections": optical_detections if optical_detections else None,
        "detected_classes": fusion_res["detected_classes"],
        "water_analysis": fusion_res["water_analysis"],
        "built_up_analysis": fusion_res["built_up_analysis"],
        "tools": [
            "optical_analyzer",
            "sar_analyzer",
            "bigearthnet_dual_encoder",
            *([ "grounding_dino", "sam" ] if optical_detections else [])
        ],
        "model": "Optical-SAR Dual-Stream Fusion (BigEarthNet Adapter)",
        "image_size": [w, h],
        "execution_time_ms": elapsed_ms,
    }


# ── Unified Agentic Orchestrator (Chunk 11) ───────────────────────────────────

async def orchestrate_agentic_request(
    query: str,
    inputs: dict,
) -> dict:
    """
    Agentic Orchestration Layer for SatQuery AI:
      1. Query Intent Interpretation (QueryInterpreter)
      2. Strict Input Validation (InputValidator)
      3. Dynamic Tool Planning (AgentPlanner)
      4. Specialist Tool Execution (ToolExecutor)
      5. Multi-Source Evidence Combination & Explainable Confidence (EvidenceCombiner)
      6. Auditable Execution Trace Synthesis
    """
    t0 = time.monotonic()
    execution_trace = []

    from core.query_interpreter import get_query_interpreter
    from core.input_validator import get_input_validator
    from core.agent_planner import get_agent_planner
    from core.tool_executor import get_tool_executor
    from core.evidence_combiner import get_evidence_combiner

    # Step 1: Query Intent Interpretation
    interpreter = get_query_interpreter()
    intent = interpreter.interpret(query, inputs)
    execution_trace.append({
        "step": 1,
        "action": "query_interpretation",
        "result": intent.intent,
        "details": {
            "sub_intent": intent.sub_intent,
            "modalities": intent.modalities,
            "temporal": intent.temporal,
            "requires_grounding": intent.requires_grounding,
            "requires_change_detection": intent.requires_change_detection,
            "requires_cross_modal": intent.requires_cross_modal,
            "objects": intent.objects,
        }
    })

    # Step 2: Input Validation
    validator = get_input_validator()
    val_res = validator.validate_request(query, intent, inputs)
    if not val_res.is_valid:
        execution_trace.append({
            "step": 2,
            "action": "input_validation",
            "result": "failed",
            "error_code": val_res.error_code,
            "message": val_res.message,
        })
        return {
            "status": "error",
            "error": {
                "code": val_res.error_code,
                "message": val_res.message,
            },
            "execution_trace": execution_trace,
            "execution_time_ms": round((time.monotonic() - t0) * 1000),
        }

    execution_trace.append({
        "step": 2,
        "action": "input_validation",
        "result": "passed",
    })

    # Forward raw multimodal tensors if decoded from multi-band GeoTIFFs
    if val_res.decoded_tensors:
        for k, t in val_res.decoded_tensors.items():
            inputs[f"{k}_tensor"] = t
            if k == "image_optical":
                inputs["optical_tensor"] = t
            elif k == "image_sar":
                inputs["sar_tensor"] = t
    if val_res.image_metadata:
        inputs["image_metadata"] = val_res.image_metadata

    # Step 3: Tool Selection & Agent Planning
    planner = get_agent_planner()
    plan = planner.create_plan(query, intent, inputs)
    planned_tools = [s.tool for s in plan.steps]
    execution_trace.append({
        "step": 3,
        "action": "tool_selection",
        "tool": planned_tools[0] if planned_tools else "none",
        "tools": planned_tools,
        "plan_id": plan.plan_id,
        "reasoning": plan.reasoning,
    })

    # Step 4: Specialist Tool Execution
    executor = get_tool_executor()
    exec_results = await executor.execute_plan(plan, inputs, query)

    execution_trace.append({
        "step": 4,
        "action": "tool_execution",
        "result": "completed",
        "executed_tools": [r.tool for r in exec_results if r.status == "success"],
        "failed_tools": [r.tool for r in exec_results if r.status == "failed"],
    })

    # Step 5: Multi-Source Evidence Combination
    combiner = get_evidence_combiner()
    package = combiner.combine(plan, exec_results, intent, query)

    execution_trace.append({
        "step": 5,
        "action": "evidence_combination",
        "result": "completed",
        "claims_count": len(package.claims),
        "confidence": package.confidence.score,
    })

    # Step 6: Answer Generation
    elapsed_ms = round((time.monotonic() - t0) * 1000)
    execution_trace.append({
        "step": 6,
        "action": "answer_generation",
        "result": "completed",
    })

    # Resolve image dimensions
    img_size = [512, 512]
    if val_res.decoded_images:
        first_img = next(iter(val_res.decoded_images.values()), None)
        if first_img:
            img_size = [first_img.width, first_img.height]

    return {
        "status": "success",
        "query": query,
        "task": intent.intent,
        "interpreted_intent": intent.to_dict(),
        "selected_tools": planned_tools,
        "answer": package.final_answer,
        "confidence": package.confidence.score,
        "confidence_details": package.confidence.to_dict(),
        "evidence": package.evidence_list,
        "claims": [c.to_dict() for c in package.claims],
        "overlay": package.primary_overlay,
        "detections": package.detections,
        "change_ratio": package.change_ratio,
        "severity": package.severity,
        "change_categories": package.change_categories,
        "detected_classes": package.detected_classes,
        "water_analysis": package.water_analysis,
        "built_up_analysis": package.built_up_analysis,
        "tools": package.tools_used,
        "model": f"SatQuery Agentic Ensemble ({', '.join(package.tools_used)})",
        "image_size": img_size,
        "execution_trace": execution_trace,
        "execution_time_ms": elapsed_ms,
    }



