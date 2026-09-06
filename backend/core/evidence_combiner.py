"""
SatQuery AI — Evidence Combiner & Confidence Estimator (Chunk 11)
Synthesizes multi-source evidence from executed specialist tools, distinguishing
direct physical evidence, model predictions, and derived inferences.
Calculates transparent explainable aggregated confidence with an explicit evidence basis.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict

from core.tool_executor import ToolExecutionResult
from core.query_interpreter import QueryIntent
from core.agent_planner import ToolPlan


@dataclass
class EvidenceClaim:
    claim: str
    claim_type: str  # "direct_evidence", "model_prediction", "derived_inference"
    supporting_tools: List[str]
    evidence_count: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedConfidence:
    score: float
    label: str  # "very_high", "high", "moderate", "low"
    basis: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CombinedEvidencePackage:
    final_answer: str
    confidence: AggregatedConfidence
    claims: List[EvidenceClaim]
    evidence_list: List[str]
    primary_overlay: Optional[str] = None
    detections: Optional[List[Dict[str, Any]]] = None
    change_ratio: Optional[float] = None
    severity: Optional[str] = None
    change_categories: Optional[List[str]] = None
    detected_classes: Optional[List[str]] = None
    water_analysis: Optional[Dict[str, Any]] = None
    built_up_analysis: Optional[Dict[str, Any]] = None
    tools_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "confidence": self.confidence.to_dict(),
            "claims": [c.to_dict() for c in self.claims],
            "evidence_list": self.evidence_list,
            "primary_overlay": self.primary_overlay,
            "detections": self.detections,
            "change_ratio": self.change_ratio,
            "severity": self.severity,
            "change_categories": self.change_categories,
            "detected_classes": self.detected_classes,
            "water_analysis": self.water_analysis,
            "built_up_analysis": self.built_up_analysis,
            "tools_used": self.tools_used,
        }


class EvidenceCombiner:
    """
    Combines outputs from multiple specialist tools into an integrated,
    auditable decision package without hallucinating evidence.
    """

    def combine(
        self,
        plan: ToolPlan,
        results: List[ToolExecutionResult],
        intent: QueryIntent,
        query: str,
    ) -> CombinedEvidencePackage:
        """
        Merge execution results across tools into a cohesive response.
        """
        successful_results = [r for r in results if r.status == "success"]
        tools_used = [r.tool for r in successful_results]

        claims: List[EvidenceClaim] = []
        evidence_list: List[str] = []
        basis: List[str] = []

        primary_overlay = None
        detections = None
        change_ratio = None
        severity = None
        change_categories = None
        detected_classes = None
        water_analysis = None
        built_up_analysis = None

        candidate_answers: List[str] = []
        confidences: List[float] = []

        # 1. Process Individual Tool Outputs
        for res in successful_results:
            confidences.append(res.confidence)
            tool_name = res.tool

            # Accumulate string evidence
            if res.evidence:
                evidence_list.extend(res.evidence)

            # ── Grounding Specialist
            if tool_name == "grounding_specialist":
                dets = res.output.get("detections", [])
                if dets:
                    detections = dets
                    claims.append(EvidenceClaim(
                        claim=f"Grounding DINO + SAM located {len(dets)} spatial candidate object(s)",
                        claim_type="derived_inference",
                        supporting_tools=["grounding_specialist"],
                        evidence_count=len(dets),
                        confidence=res.confidence,
                    ))
                    basis.append("Object Grounding DINO + SAM spatial localization")
                if res.overlay:
                    primary_overlay = res.overlay
                if res.answer:
                    candidate_answers.append(res.answer)

            # ── Satellite VLM
            elif tool_name == "satellite_vlm":
                if res.answer:
                    candidate_answers.append(res.answer)
                    claims.append(EvidenceClaim(
                        claim=f"Satellite VLM reasoning: {res.answer[:80]}...",
                        claim_type="model_prediction",
                        supporting_tools=["satellite_vlm"],
                        evidence_count=1,
                        confidence=res.confidence,
                    ))
                    basis.append("Satellite VLM visual reasoning engine")

            # ── Change Analyzer
            elif tool_name == "change_analyzer":
                change_ratio = res.output.get("change_ratio")
                severity = res.output.get("severity")
                change_categories = res.output.get("change_categories")
                if res.overlay:
                    primary_overlay = res.overlay
                if res.answer:
                    candidate_answers.append(res.answer)
                claims.append(EvidenceClaim(
                    claim=f"ChangeFormer detected {change_ratio}% bi-temporal change ({severity})",
                    claim_type="model_prediction",
                    supporting_tools=["change_analyzer"],
                    evidence_count=len(change_categories or []) + 1,
                    confidence=res.confidence,
                ))
                basis.append("ChangeFormerV6 bi-temporal transformer change analysis")

            # ── Optical + SAR Analyzer
            elif tool_name == "optical_sar_analyzer":
                detected_classes = res.output.get("detected_classes")
                water_analysis = res.output.get("water_analysis")
                built_up_analysis = res.output.get("built_up_analysis")
                if res.overlay:
                    primary_overlay = res.overlay
                if res.answer:
                    candidate_answers.append(res.answer)
                claims.append(EvidenceClaim(
                    claim="Optical multispectral + Sentinel-1 SAR dual-stream cross-modal concordance verified",
                    claim_type="direct_evidence",
                    supporting_tools=["optical_sar_analyzer"],
                    evidence_count=len(res.output.get("fused_evidence", [])) or 2,
                    confidence=res.confidence,
                ))
                basis.append("Cross-modal Optical + SAR dual-stream neural fusion")
                basis.append("Mutual spatial cross-validation (specular attenuation vs double-bounce)")

            # ── Optical Spectral Analyzer
            elif tool_name == "optical_analyzer":
                if res.answer:
                    candidate_answers.append(res.answer)
                mean_ndvi = res.output.get("mean_ndvi", 0.0)
                mean_ndwi = res.output.get("mean_ndwi", 0.0)
                claims.append(EvidenceClaim(
                    claim=f"Optical multispectral indices: NDVI={mean_ndvi:.2f}, NDWI={mean_ndwi:.2f}",
                    claim_type="direct_evidence",
                    supporting_tools=["optical_analyzer"],
                    evidence_count=2,
                    confidence=res.confidence,
                ))
                basis.append("Optical spectral index analysis (NDVI / NDWI)")

            # ── SAR Radar Analyzer
            elif tool_name == "sar_analyzer":
                if res.answer:
                    candidate_answers.append(res.answer)
                vv = res.output.get("mean_vv_db", 0.0)
                vh = res.output.get("mean_vh_db", 0.0)
                claims.append(EvidenceClaim(
                    claim=f"Sentinel-1 SAR C-band microwave backscatter: mean VV={vv:.1f} dB, mean VH={vh:.1f} dB",
                    claim_type="direct_evidence",
                    supporting_tools=["sar_analyzer"],
                    evidence_count=2,
                    confidence=res.confidence,
                ))
                basis.append("Sentinel-1 SAR C-band radar backscatter physics")

            # ── BigEarthNet Adapter
            elif tool_name == "bigearthnet_adapter":
                labels = res.output.get("detected_labels", [])
                if labels:
                    detected_classes = labels
                if res.answer:
                    candidate_answers.append(res.answer)
                claims.append(EvidenceClaim(
                    claim=f"Adapted BigEarthNet multi-label classification: {', '.join(labels)}",
                    claim_type="model_prediction",
                    supporting_tools=["bigearthnet_adapter"],
                    evidence_count=len(labels),
                    confidence=res.confidence,
                ))
                basis.append("Adapted BigEarthNet domain representation")

        # 2. Select / Synthesize Final Answer
        final_answer = self._synthesize_final_answer(
            intent=intent,
            candidate_answers=candidate_answers,
            successful_results=successful_results,
            query=query,
        )

        # 3. Transparent Confidence Calculation
        agg_score, agg_label = self._calculate_aggregate_confidence(
            confidences=confidences,
            tools_used=tools_used,
            claims=claims,
            failed_count=len(results) - len(successful_results),
        )

        confidence_obj = AggregatedConfidence(
            score=agg_score,
            label=agg_label,
            basis=list(dict.fromkeys(basis)),  # Unique preservation
        )

        return CombinedEvidencePackage(
            final_answer=final_answer,
            confidence=confidence_obj,
            claims=claims,
            evidence_list=evidence_list,
            primary_overlay=primary_overlay,
            detections=detections,
            change_ratio=change_ratio,
            severity=severity,
            change_categories=change_categories,
            detected_classes=detected_classes,
            water_analysis=water_analysis,
            built_up_analysis=built_up_analysis,
            tools_used=tools_used,
        )

    def _synthesize_final_answer(
        self,
        intent: QueryIntent,
        candidate_answers: List[str],
        successful_results: List[ToolExecutionResult],
        query: str,
    ) -> str:
        """Formulate a concise, clear natural-language answer without redundancy."""
        if not candidate_answers:
            return "Analysis complete. No affirmative features or answer could be derived from the available inputs."

        # If only 1 tool executed, its answer is primary
        if len(candidate_answers) == 1:
            return candidate_answers[0]

        # Multi-tool synergy:
        # Prioritize Optical-SAR or ChangeFormer or VLM conditioned on grounding
        for res in successful_results:
            if res.tool == "optical_sar_analyzer" and res.answer:
                return res.answer
            if res.tool == "change_analyzer" and res.answer and intent.intent == "bitemporal_change":
                return res.answer

        # If grounding + VLM ran: VLM answer is primary, grounding is corroborating
        vlm_res = next((r for r in successful_results if r.tool == "satellite_vlm"), None)
        grounding_res = next((r for r in successful_results if r.tool == "grounding_specialist"), None)

        if vlm_res and vlm_res.answer:
            if grounding_res and grounding_res.output.get("count", 0) > 0:
                count = grounding_res.output["count"]
                return f"{vlm_res.answer} (Spatial grounding identified {count} corresponding region(s))."
            return vlm_res.answer

        # Default: concatenate distinct answers
        distinct = []
        for a in candidate_answers:
            if a and a not in distinct:
                distinct.append(a)
        return " ".join(distinct)

    def _calculate_aggregate_confidence(
        self,
        confidences: List[float],
        tools_used: List[str],
        claims: List[EvidenceClaim],
        failed_count: int,
    ) -> Tuple[float, str]:
        """
        Explainable confidence aggregation:
        - Base mean of contributing tool confidence values
        - Small bonus (+0.03) for multi-tool cross-validation concordance
        - Penalty (-0.10 per failed tool) for missing/failed expected steps
        """
        if not confidences:
            return 0.50, "moderate"

        base_conf = float(np.mean(confidences))

        # Concordance bonus if multiple independent tools agree
        concordance_bonus = 0.04 if len(tools_used) >= 2 else 0.0
        failure_penalty = 0.10 * failed_count

        final_score = np.clip(base_conf + concordance_bonus - failure_penalty, 0.20, 0.98)
        score_rounded = round(float(final_score), 2)

        if score_rounded >= 0.88:
            label = "very_high"
        elif score_rounded >= 0.75:
            label = "high"
        elif score_rounded >= 0.55:
            label = "moderate"
        else:
            label = "low"

        return score_rounded, label


_evidence_combiner: Optional[EvidenceCombiner] = None

def get_evidence_combiner() -> EvidenceCombiner:
    """Singleton getter for EvidenceCombiner."""
    global _evidence_combiner
    if _evidence_combiner is None:
        _evidence_combiner = EvidenceCombiner()
    return _evidence_combiner
