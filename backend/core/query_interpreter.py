"""
SatQuery AI — Query Intent Interpreter (Chunk 11)
Analyzes natural-language satellite queries to extract structured intent,
required modalities, temporal modes, candidate target objects, and specialist tool requirements.
"""
from __future__ import annotations
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class QueryIntent:
    """Structured semantic representation of a user's natural-language satellite query."""
    intent: str
    sub_intent: str
    modalities: List[str]
    temporal: str
    requires_grounding: bool
    requires_change_detection: bool
    requires_cross_modal: bool
    requires_optical_analysis: bool
    requires_sar_analysis: bool
    objects: List[str] = field(default_factory=list)
    confidence: float = 0.95
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Regex Patterns for Intent Disambiguation ──────────────────────────────────

_CROSS_MODAL_PATTERNS = [
    r"\b(optical\s+and\s+sar|sar\s+and\s+optical|optical\s+and\s+radar|radar\s+and\s+optical)\b",
    r"\b(cross[-_ ]modal|multimodal\s+fusion|fused\s+analysis|both\s+modalities)\b",
    r"\b(optical\s+and\s+sar\s+images\s+together|optical\s+and\s+radar\s+images\s+together)\b",
    r"\b(compare\s+the\s+optical\s+and\s+sar|optical\s+and\s+sar\s+evidence)\b",
    r"\b(together\s+to\s+identify\s+built[- ]up\s+and\s+water)\b",
]

_CHANGE_PATTERNS = [
    r"\b(change|changed|changes|changing|difference|differed)\b",
    r"\b(between\s+(these\s+)?(two\s+)?dates|between\s+(the\s+)?two\s+images|over\s+time|before\s+and\s+after)\b",
    r"\b(increased|decreased|increase\s+or\s+decrease|expansion|reduction|shrinkage|growth)\b",
    r"\b(new\s+construction|demolished|cleared|transitioned\s+to)\b",
    r"\b(change\s+ratio|change\s+map|where\s+did\s+the\s+change\s+occur)\b",
]

_GROUNDING_PATTERNS = [
    r"\b(highlight|locate|find|where\s+is|where\s+are|segment|point\s+out|detect|bounding\s+box|bbox)\b",
    r"\b(mark|show\s+me\s+the|draw\s+box|pinpoint)\b",
]

_OPTICAL_SPECIFIC_PATTERNS = [
    r"\b(ndvi|ndwi|multispectral|vegetation\s+index|water\s+index|optical\s+bands?|nir|b02|b03|b04|b08)\b",
    r"\b(estimate\s+vegetation\s+using\s+(the\s+)?optical|optical\s+image\s+only|spectral\s+reflectance)\b",
]

_SAR_SPECIFIC_PATTERNS = [
    r"\b(vv|vh|backscatter|radar|c[- ]band|grd|microwave|specular|double[- ]bounce|dihedral)\b",
    r"\b(use\s+sar\s+to|sar\s+image\s+only|radar\s+image|sar\s+polarization)\b",
]

_DESCRIPTION_PATTERNS = [
    r"\b(describe|caption|overview|summarize|what\s+is\s+visible|what\s+do\s+you\s+see|tell\s+me\s+about)\b",
    r"\b(describe\s+the\s+land[- ]cover|major\s+objects\s+visible)\b",
]

_OBJECT_KEYWORDS = [
    "water body", "water", "lake", "river", "reservoir", "ocean", "wetland",
    "building", "buildings", "house", "houses", "structure", "structures", "built-up", "urban",
    "road", "roads", "highway", "runway", "airport", "bridge",
    "tree", "trees", "forest", "vegetation", "grassland", "crop", "cropland", "field",
    "car", "cars", "vehicle", "vehicles", "ship", "ships", "vessel", "vessels",
    "playground", "industrial unit", "commercial unit", "solar panel",
]


class QueryInterpreter:
    """
    Intelligent Query Interpreter for SatQuery AI.
    Disambiguates single-image, bi-temporal, optical-SAR cross-modal,
    spectral, radar, and complex multi-tool intent from raw text queries.
    """

    def interpret(
        self,
        query: str,
        input_context: Optional[Dict[str, Any]] = None
    ) -> QueryIntent:
        """
        Parse query and optional input availability context into a structured QueryIntent.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return QueryIntent(
                intent="unknown",
                sub_intent="empty_query",
                modalities=[],
                temporal="unknown",
                requires_grounding=False,
                requires_change_detection=False,
                requires_cross_modal=False,
                requires_optical_analysis=False,
                requires_sar_analysis=False,
                confidence=0.0,
                reasoning="Empty or whitespace query provided.",
            )

        q_clean = query.strip()
        q_lower = q_clean.lower()
        inputs = input_context or {}

        # Extract referenced objects
        extracted_objects = []
        for obj in _OBJECT_KEYWORDS:
            if re.search(r"\b" + re.escape(obj) + r"\b", q_lower):
                extracted_objects.append(obj)

        # Check for multi-condition / complex multi-tool queries
        has_change_mention = any(re.search(pat, q_lower) for pat in _CHANGE_PATTERNS)
        has_grounding_mention = any(re.search(pat, q_lower) for pat in _GROUNDING_PATTERNS)
        has_crossmodal_mention = any(re.search(pat, q_lower) for pat in _CROSS_MODAL_PATTERNS)
        has_optical_mention = any(re.search(pat, q_lower) for pat in _OPTICAL_SPECIFIC_PATTERNS)
        has_sar_mention = any(re.search(pat, q_lower) for pat in _SAR_SPECIFIC_PATTERNS)
        has_description_mention = any(re.search(pat, q_lower) for pat in _DESCRIPTION_PATTERNS)

        # Count active intent domains
        active_domains = sum([
            has_change_mention,
            has_grounding_mention,
            has_crossmodal_mention,
            (has_optical_mention or has_sar_mention),
        ])

        # ── H. Complex Multi-Tool Query ───────────────────────────────────────
        if active_domains >= 2 and ("and" in q_lower or "," in q_lower):
            # Example: "Find the buildings, determine whether built-up area increased between the two dates, and explain the evidence using optical and SAR."
            return QueryIntent(
                intent="complex_multitool",
                sub_intent="compound_multi_stage",
                modalities=["optical", "sar"] if (has_crossmodal_mention or has_sar_mention) else ["optical"],
                temporal="bitemporal" if has_change_mention else "single",
                requires_grounding=has_grounding_mention,
                requires_change_detection=has_change_mention,
                requires_cross_modal=has_crossmodal_mention,
                requires_optical_analysis=has_optical_mention,
                requires_sar_analysis=has_sar_mention,
                objects=extracted_objects,
                confidence=0.96,
                reasoning="Query contains compound multi-clause goals combining change detection, spatial localization, and/or cross-modal fusion.",
            )

        # ── E. Optical + SAR Cross-Modal Query ─────────────────────────────────
        if has_crossmodal_mention or (has_optical_mention and has_sar_mention) or inputs.get("modality") == "optical_sar":
            sub = "built_up_and_water" if ("built-up" in q_lower or "built up" in q_lower) and "water" in q_lower else "sensor_comparison"
            return QueryIntent(
                intent="optical_sar_crossmodal",
                sub_intent=sub,
                modalities=["optical", "sar"],
                temporal="single",
                requires_grounding=has_grounding_mention,
                requires_change_detection=False,
                requires_cross_modal=True,
                requires_optical_analysis=True,
                requires_sar_analysis=True,
                objects=extracted_objects,
                confidence=0.95,
                reasoning="Query explicitly references joint optical and SAR sensor synergy or complementary cross-modal analysis.",
            )

        # ── D. Bi-Temporal Change Detection & Change-VQA ──────────────────────
        if has_change_mention or inputs.get("has_two_dates") or inputs.get("image_t2"):
            sub = "change_vqa" if ("?" in q_lower or any(w in q_lower for w in ["has", "did", "is", "are", "what", "how"])) else "change_detection"
            return QueryIntent(
                intent="bitemporal_change",
                sub_intent=sub,
                modalities=["optical"],
                temporal="bitemporal",
                requires_grounding=False,
                requires_change_detection=True,
                requires_cross_modal=False,
                requires_optical_analysis=False,
                requires_sar_analysis=False,
                objects=extracted_objects,
                confidence=0.94,
                reasoning="Query requests detection, quantification, or question answering regarding bi-temporal changes over time.",
            )

        # ── F. Optical-Specific Spectral Query ────────────────────────────────
        if has_optical_mention and not has_sar_mention:
            return QueryIntent(
                intent="optical_specific",
                sub_intent="spectral_index",
                modalities=["optical"],
                temporal="single",
                requires_grounding=False,
                requires_change_detection=False,
                requires_cross_modal=False,
                requires_optical_analysis=True,
                requires_sar_analysis=False,
                objects=extracted_objects,
                confidence=0.92,
                reasoning="Query specifically targets optical multispectral indices (NDVI/NDWI) or spectral bands.",
            )

        # ── G. SAR-Specific Radar Query ───────────────────────────────────────
        if has_sar_mention and not has_optical_mention:
            return QueryIntent(
                intent="sar_specific",
                sub_intent="polarimetric_backscatter",
                modalities=["sar"],
                temporal="single",
                requires_grounding=False,
                requires_change_detection=False,
                requires_cross_modal=False,
                requires_optical_analysis=False,
                requires_sar_analysis=True,
                objects=extracted_objects,
                confidence=0.92,
                reasoning="Query specifically targets Sentinel-1 SAR C-band microwave backscatter (VV/VH dB) or surface roughness.",
            )

        # ── C. Object Grounding / Spatial Localization ────────────────────────
        if has_grounding_mention:
            return QueryIntent(
                intent="object_grounding",
                sub_intent="spatial_localization",
                modalities=["optical"],
                temporal="single",
                requires_grounding=True,
                requires_change_detection=False,
                requires_cross_modal=False,
                requires_optical_analysis=False,
                requires_sar_analysis=False,
                objects=extracted_objects,
                confidence=0.93,
                reasoning="Query instructs spatial localization, segmentation, or bounding box detection for target scene features.",
            )

        # ── A. Single-Image Scene Description ─────────────────────────────────
        if has_description_mention or "describe" in q_lower or "caption" in q_lower:
            return QueryIntent(
                intent="single_image_description",
                sub_intent="scene_description",
                modalities=["optical"],
                temporal="single",
                requires_grounding=False,
                requires_change_detection=False,
                requires_cross_modal=False,
                requires_optical_analysis=False,
                requires_sar_analysis=False,
                objects=extracted_objects,
                confidence=0.91,
                reasoning="Query requests a comprehensive descriptive overview or natural-language caption of the satellite scene.",
            )

        # ── B. Single-Image VQA (Default Single-Image Fallback) ────────────────
        sub = "count" if any(w in q_lower for w in ["how many", "count", "number of"]) else "presence"
        req_grounding = sub == "count"  # Counting benefits from object grounding bounding boxes
        return QueryIntent(
            intent="single_image_vqa",
            sub_intent=sub,
            modalities=["optical"],
            temporal="single",
            requires_grounding=req_grounding,
            requires_change_detection=False,
            requires_cross_modal=False,
            requires_optical_analysis=False,
            requires_sar_analysis=False,
            objects=extracted_objects,
            confidence=0.90,
            reasoning="Query poses a specific question regarding presence, counts, or attributes within a single satellite image.",
        )


_query_interpreter: Optional[QueryInterpreter] = None

def get_query_interpreter() -> QueryInterpreter:
    """Singleton getter for QueryInterpreter."""
    global _query_interpreter
    if _query_interpreter is None:
        _query_interpreter = QueryInterpreter()
    return _query_interpreter
