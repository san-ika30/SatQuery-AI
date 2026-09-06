"""
SatQuery AI — Input Validator (Chunk 11)
Validates query and image inputs against intent and tool requirements.
Prevents execution when required modalities, temporal pairs, or payloads are missing.
"""
from __future__ import annotations
import base64
import io
from typing import Optional, Dict, Any, Tuple
from PIL import Image
from dataclasses import dataclass

from core.query_interpreter import QueryIntent


@dataclass
class ValidationResult:
    is_valid: bool
    status: str = "valid"
    error_code: Optional[str] = None
    message: Optional[str] = None
    decoded_images: Dict[str, Image.Image] = None
    decoded_tensors: Dict[str, Any] = None
    image_metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        if self.is_valid:
            return {"status": "valid"}
        return {
            "status": "input_error",
            "error_code": self.error_code,
            "message": self.message,
        }


class InputValidator:
    """
    Strict input validation layer for SatQuery AI.
    Enforces SIH input constraints before planner/tool execution.
    """

    def validate_request(
        self,
        query: str,
        intent: QueryIntent,
        inputs: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate natural-language query, provided image payloads, and temporal/modality alignment.
        """
        # 1. Query Validation
        if not query or not isinstance(query, str) or not query.strip():
            return ValidationResult(
                is_valid=False,
                status="input_error",
                error_code="EMPTY_QUESTION",
                message="Question text cannot be empty.",
            )

        # 2. Extract potential image payloads
        img_single = inputs.get("image") or inputs.get("image_t1") or inputs.get("image_optical")
        img_t1 = inputs.get("image_t1")
        img_t2 = inputs.get("image_t2")
        img_optical = inputs.get("image_optical") or inputs.get("image") or inputs.get("image_t1")
        img_sar = inputs.get("image_sar")

        # 3. Check Temporal Pair Completeness
        if intent.requires_change_detection or intent.temporal == "bitemporal":
            has_two_images = bool(img_t1 and img_t2)
            if not has_two_images:
                return ValidationResult(
                    is_valid=False,
                    status="input_error",
                    error_code="MISSING_TEMPORAL_IMAGE",
                    message="This query requires two images from different dates.",
                )

        # 4. Check Optical + SAR Modality Completeness
        if intent.requires_cross_modal or (intent.intent == "optical_sar_crossmodal"):
            if not img_sar:
                return ValidationResult(
                    is_valid=False,
                    status="input_error",
                    error_code="MISSING_SAR_IMAGE",
                    message="This query requires both optical and SAR imagery.",
                )
            if not img_optical:
                return ValidationResult(
                    is_valid=False,
                    status="input_error",
                    error_code="INVALID_OPTICAL_PAYLOAD",
                    message="This query requires an optical image payload.",
                )

        # 5. Check Single Image Presence
        if not img_single and not img_t1 and not img_optical:
            return ValidationResult(
                is_valid=False,
                status="input_error",
                error_code="INVALID_IMAGE_PAYLOAD",
                message="Base64 image input is empty or invalid.",
            )

        # 6. Verify Base64 Decoding & PIL Image Validity
        decoded: Dict[str, Image.Image] = {}
        tensors: Dict[str, Any] = {}
        meta_dict: Dict[str, Any] = {}

        modality_hint = inputs.get("modality")

        for key, payload in [
            ("image", img_single),
            ("image_t1", img_t1),
            ("image_t2", img_t2),
            ("image_optical", img_optical),
            ("image_sar", img_sar),
        ]:
            if payload is not None:
                key_hint = modality_hint
                if key == "image_optical":
                    key_hint = "optical"
                elif key == "image_sar":
                    key_hint = "sar"
                elif key in ("image_t1", "image_t2"):
                    key_hint = "bitemporal"
                elif intent.requires_cross_modal and key == "image":
                    key_hint = "optical"

                is_valid_img, pil_img, err, raw_tensor, meta = self._decode_image(
                    payload, key=key, modality_hint=key_hint
                )
                if not is_valid_img:
                    return ValidationResult(
                        is_valid=False,
                        status="input_error",
                        error_code="INVALID_IMAGE_FORMAT",
                        message=f"Failed to decode image file for '{key}': {err}",
                    )
                decoded[key] = pil_img
                if raw_tensor is not None:
                    tensors[key] = raw_tensor
                if meta is not None:
                    meta_dict[key] = meta

        return ValidationResult(
            is_valid=True,
            status="valid",
            decoded_images=decoded,
            decoded_tensors=tensors,
            image_metadata=meta_dict,
        )

    def _decode_image(
        self,
        payload: Any,
        key: Optional[str] = None,
        modality_hint: Optional[str] = None,
    ) -> Tuple[bool, Optional[Image.Image], Optional[str], Optional[Any], Optional[dict]]:
        if isinstance(payload, Image.Image):
            return True, payload, None, None, {"format": "pil"}

        if not isinstance(payload, (str, bytes)):
            return False, None, f"Unsupported image payload type: {type(payload)}", None, None

        try:
            if isinstance(payload, str):
                clean_b64 = payload.strip()
                if "," in clean_b64:
                    clean_b64 = clean_b64.split(",", 1)[1]
                raw_bytes = base64.b64decode(clean_b64)
            else:
                raw_bytes = payload

            if not raw_bytes or len(raw_bytes) < 8:
                return False, None, "Decoded byte stream is empty or too short.", None, None

            # Detect TIFF magic bytes
            is_tiff = (
                len(raw_bytes) >= 4 and (
                    raw_bytes[:4] in (b"II*\x00", b"MM\x00*") or
                    raw_bytes[:2] in (b"II", b"MM")
                )
            )

            if is_tiff:
                from core.preprocessor import decode_multimodal_geotiff
                pil_img, raw_tensor, meta = decode_multimodal_geotiff(
                    raw_bytes,
                    modality_hint=modality_hint,
                )
                if pil_img.width <= 0 or pil_img.height <= 0:
                    return False, None, "Image has invalid zero or negative dimensions.", None, None
                return True, pil_img, None, raw_tensor, meta

            # Standard non-TIFF raster path (PNG, JPG)
            img = Image.open(io.BytesIO(raw_bytes))
            img.verify()  # Verify image integrity
            # Reopen after verify because verify can close buffer
            img = Image.open(io.BytesIO(raw_bytes))
            if img.width <= 0 or img.height <= 0:
                return False, None, "Image has invalid zero or negative dimensions.", None, None

            return True, img, None, None, {"format": "standard_raster"}
        except Exception as exc:
            return False, None, str(exc), None, None


_input_validator: Optional[InputValidator] = None

def get_input_validator() -> InputValidator:
    """Singleton getter for InputValidator."""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator
