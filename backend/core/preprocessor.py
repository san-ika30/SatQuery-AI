"""
SatQuery AI — GeoTIFF/TIFF Preprocessor
Handles band extraction, normalization, modality detection, and PNG conversion.
"""
from __future__ import annotations
import io
import os
import tempfile
import numpy as np
import structlog
from pathlib import Path
from typing import Optional
from PIL import Image

try:
    import rasterio
    from rasterio.enums import ColorInterp
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

import torch
from schemas.models import ImageMetadata, Modality

log = structlog.get_logger()


def decode_multimodal_geotiff(
    raw_bytes: bytes,
    modality_hint: Optional[str] = None,
    filename: str = "",
) -> tuple[Image.Image, Optional[torch.Tensor], dict]:
    """
    Decodes multi-band GeoTIFF bytes into:
      1. RGB PIL Image (for VLM, Grounding, display)
      2. Raw tensor (4-channel for S2, 2-channel for S1) when applicable
      3. Metadata dictionary describing band count, layout, and modality

    Supports:
      - 3-band RGB TIFF/GeoTIFF: normal 8-bit RGB PIL image
      - 4-band Sentinel-2 GeoTIFF [B02, B03, B04, B08]:
          R=B04, G=B03, B=B02 for RGB visualization, raw (4, H, W) tensor for OpticalSARAnalyzer
      - 2-band Sentinel-1 SAR GeoTIFF [VV, VH]:
          false-color RGB composite, raw (2, H, W) tensor for OpticalSARAnalyzer
      - Bi-temporal pairs of supported GeoTIFFs
    """
    is_tiff = (
        (len(raw_bytes) >= 4 and (raw_bytes[:4] in (b"II*\x00", b"MM\x00*") or raw_bytes[:2] in (b"II", b"MM")))
        or any(filename.lower().endswith(ext) for ext in [".tif", ".tiff", ".geotiff"])
    )
    if not is_tiff or not TIFFFILE_AVAILABLE:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return pil_img, None, {"format": "standard_raster", "band_count": 3}

    try:
        arr = tifffile.imread(io.BytesIO(raw_bytes))
    except Exception as exc:
        log.warning("tifffile_decode_fallback_pillow", error=str(exc))
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return pil_img, None, {"format": "pillow_fallback", "error": str(exc)}

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] not in (1, 2, 3, 4) and arr.shape[2] in (1, 2, 3, 4):
            arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported TIFF array dimension: {arr.ndim}")

    channels, height, width = arr.shape

    if channels > 4:
        raise ValueError(
            f"Unsupported multi-band GeoTIFF layout with {channels} bands. "
            "Supported band counts: 1 (single-band), 2 (Sentinel-1 VV/VH), 3 (RGB), 4 (Sentinel-2 B02/B03/B04/B08)."
        )

    hint_lower = (modality_hint or "").lower()
    fn_lower = filename.lower()

    is_s2 = (
        any(k in hint_lower for k in ["s2", "sentinel2", "sentinel-2", "optical", "multispectral", "bitemporal"])
        or any(k in fn_lower for k in ["s2", "sentinel2", "sentinel-2", "optical", "b02", "b04", "b08"])
        or hint_lower in ("image_optical", "optical")
    )

    is_s1 = (
        any(k in hint_lower for k in ["s1", "sentinel1", "sentinel-1", "sar", "radar"])
        or any(k in fn_lower for k in ["s1", "sentinel1", "sentinel-1", "sar", "vv", "vh"])
        or hint_lower in ("image_sar", "sar")
    )

    # 1. 4-band Sentinel-2 vs Generic 4-band
    if channels == 4:
        if is_s2:
            # Sentinel-2: [B02(Blue), B03(Green), B04(Red), B08(NIR)]
            # RGB visualization: R = B04 (idx 2), G = B03 (idx 1), B = B02 (idx 0)
            r_vis = normalize_band(arr[2])
            g_vis = normalize_band(arr[1])
            b_vis = normalize_band(arr[0])
            pil_img = Image.fromarray(np.stack([r_vis, g_vis, b_vis], axis=-1), mode="RGB")

            # Raw 4-channel tensor for OpticalSARAnalyzer
            if np.issubdtype(arr.dtype, np.integer):
                s2_norm = np.clip(arr.astype(np.float32) / 3000.0, 0.0, 1.0)
            else:
                s2_norm = np.clip(arr.astype(np.float32), 0.0, 1.0)
            raw_tensor = torch.tensor(s2_norm, dtype=torch.float32)
            meta = {
                "format": "sentinel2_geotiff",
                "band_count": 4,
                "layout": ["B02", "B03", "B04", "B08"],
                "dtype": str(arr.dtype),
                "modality": "optical",
                "shape": list(arr.shape),
            }
            return pil_img, raw_tensor, meta
        else:
            # Generic 4-band TIFF (e.g. RGBA) - NOT treated as S2
            r_vis = normalize_band(arr[0])
            g_vis = normalize_band(arr[1])
            b_vis = normalize_band(arr[2])
            pil_img = Image.fromarray(np.stack([r_vis, g_vis, b_vis], axis=-1), mode="RGB")
            meta = {
                "format": "generic_4band_tiff",
                "band_count": 4,
                "layout": "RGBA",
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            }
            return pil_img, None, meta

    # 2. 2-band Sentinel-1 SAR vs Generic 2-band
    elif channels == 2:
        if is_s1:
            # Sentinel-1 SAR: [VV, VH]
            vv_raw = arr[0].astype(np.float32)
            vh_raw = arr[1].astype(np.float32)

            if np.mean(vv_raw) < 15.0 and np.mean(vv_raw) < 0.0:  # in dB
                vv_db = vv_raw
                vh_db = vh_raw
            else:
                vv_db = 10.0 * np.log10(np.maximum(vv_raw, 1e-5))
                vh_db = 10.0 * np.log10(np.maximum(vh_raw, 1e-5))

            vv_norm = (vv_db - (-15.0)) / 10.0
            vh_norm = (vh_db - (-20.0)) / 10.0
            s1_norm = np.stack([vv_norm, vh_norm], axis=0).astype(np.float32)
            raw_tensor = torch.tensor(s1_norm, dtype=torch.float32)

            # False-color composite: VV, VH, difference
            vv_u8 = normalize_band(vv_raw)
            vh_u8 = normalize_band(vh_raw)
            diff = np.clip(vv_u8.astype(np.int16) - vh_u8.astype(np.int16) + 128, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(np.stack([vv_u8, vh_u8, diff], axis=-1), mode="RGB")
            meta = {
                "format": "sentinel1_geotiff",
                "band_count": 2,
                "layout": ["VV", "VH"],
                "dtype": str(arr.dtype),
                "modality": "sar",
                "shape": list(arr.shape),
            }
            return pil_img, raw_tensor, meta
        else:
            b0 = normalize_band(arr[0])
            b1 = normalize_band(arr[1])
            pil_img = Image.fromarray(np.stack([b0, b1, b0], axis=-1), mode="RGB")
            meta = {
                "format": "generic_2band_tiff",
                "band_count": 2,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            }
            return pil_img, None, meta

    # 3. 3-band RGB TIFF
    elif channels == 3:
        r_vis = normalize_band(arr[0])
        g_vis = normalize_band(arr[1])
        b_vis = normalize_band(arr[2])
        pil_img = Image.fromarray(np.stack([r_vis, g_vis, b_vis], axis=-1), mode="RGB")
        meta = {
            "format": "rgb_geotiff",
            "band_count": 3,
            "layout": "RGB",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
        return pil_img, None, meta

    # 4. 1-band Grayscale / Single-pol TIFF
    else:
        b0 = normalize_band(arr[0])
        pil_img = Image.fromarray(np.stack([b0, b0, b0], axis=-1), mode="RGB")
        meta = {
            "format": "single_band_tiff",
            "band_count": 1,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
        return pil_img, None, meta


class PreprocessingError(Exception):
    pass


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"}


def detect_modality(band_count: int, filename: str) -> Modality:
    """Heuristically detect image modality from filename and band count."""
    name_lower = filename.lower()
    if any(kw in name_lower for kw in ["sar", "s1", "sentinel1", "sentinel-1", "risat"]):
        return Modality.sar
    if any(kw in name_lower for kw in ["s2", "sentinel2", "sentinel-2", "cartosat", "optical"]):
        return Modality.multispectral if band_count > 4 else Modality.optical
    # Fallback by band count
    if band_count == 1:
        return Modality.sar  # single-band is likely SAR
    if band_count <= 4:
        return Modality.optical
    return Modality.multispectral


def normalize_band(band: np.ndarray) -> np.ndarray:
    """Normalize a single band to [0, 255] uint8."""
    band = band.astype(np.float32)
    p2, p98 = np.percentile(band[band != 0], (2, 98)) if band.any() else (0, 1)
    band = np.clip(band, p2, p98)
    rng = p98 - p2
    if rng == 0:
        return np.zeros_like(band, dtype=np.uint8)
    band = ((band - p2) / rng * 255).astype(np.uint8)
    return band


def _process_with_pillow(
    file_bytes: bytes,
    filename: str,
) -> tuple[bytes, ImageMetadata]:
    """
    Fast-path for standard raster images (JPG, PNG) using Pillow only.
    Used when rasterio is not installed or for non-GeoTIFF inputs.
    """
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise PreprocessingError(f"Pillow could not open '{filename}': {exc}")

    width, height = pil_img.size
    band_count = 3  # Always RGB after convert
    modality = detect_modality(band_count, filename)

    # Resize if too large
    max_side = 1024
    if max(width, height) > max_side:
        pil_img.thumbnail((max_side, max_side), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    metadata = ImageMetadata(
        filename=filename,
        modality=modality,
        width=width,
        height=height,
        band_count=band_count,
        dtype="uint8",
        crs=None,
        bounds=None,
    )

    log.info("preprocessed_image_pillow",
             filename=filename,
             modality=modality.value,
             size=f"{width}x{height}")
    return png_bytes, metadata


def geotiff_to_rgb_png(
    file_bytes: bytes,
    filename: str,
    rgb_bands: Optional[list[int]] = None,
) -> tuple[bytes, ImageMetadata]:
    """
    Convert a GeoTIFF (or any raster) to an RGB PNG for VLM inference.

    - JPG/PNG files: handled via Pillow (no rasterio needed)
    - GeoTIFF/TIFF:  handled via rasterio (falls back to Pillow if unavailable)

    Returns:
        (png_bytes, ImageMetadata)
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise PreprocessingError(f"Unsupported file format: {ext}")

    # ── GeoTIFF decoding via tifffile (in-memory multi-band support) ──────────
    if ext in {".tif", ".tiff", ".geotiff"} and TIFFFILE_AVAILABLE:
        try:
            pil_img, _, meta = decode_multimodal_geotiff(file_bytes, filename=filename)
            width, height = pil_img.size
            modality = detect_modality(meta.get("band_count", 3), filename)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            metadata = ImageMetadata(
                filename=filename,
                modality=modality,
                width=width,
                height=height,
                band_count=meta.get("band_count", 3),
                dtype=meta.get("dtype", "uint16"),
                crs=None,
                bounds=None,
            )
            log.info("preprocessed_image",
                     filename=filename,
                     modality=modality.value,
                     bands=meta.get("band_count", 3),
                     size=f"{width}x{height}")
            return png_bytes, metadata
        except Exception as exc:
            log.warning("geotiff_tifffile_decode_failed", error=str(exc))

    # ── Fast path: standard image formats via Pillow (no rasterio required) ──
    if ext in {".png", ".jpg", ".jpeg"} or not RASTERIO_AVAILABLE:
        return _process_with_pillow(file_bytes, filename)

    # ── Full GeoTIFF path via rasterio ────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path) as src:
            band_count = src.count
            width = src.width
            height = src.height
            dtype = str(src.dtypes[0])
            crs = str(src.crs) if src.crs else None
            bounds = list(src.bounds) if src.bounds else None

            modality = detect_modality(band_count, filename)

            # Select RGB bands
            if rgb_bands is None:
                if band_count >= 3:
                    if modality == Modality.multispectral and band_count >= 4:
                        rgb_bands = [4, 3, 2]
                    else:
                        rgb_bands = [1, 2, 3]
                elif band_count == 2:
                    rgb_bands = [1, 2, 1]  # pseudo-RGB
                else:
                    rgb_bands = [1, 1, 1]  # grayscale SAR → pseudo-RGB

            channels = []
            for b in rgb_bands:
                band_idx = min(b, band_count)
                data = src.read(band_idx)
                channels.append(normalize_band(data))

            rgb = np.stack(channels, axis=-1)  # H x W x 3
            pil_img = Image.fromarray(rgb, mode="RGB")

            # Resize if too large
            max_side = 1024
            if max(pil_img.width, pil_img.height) > max_side:
                pil_img.thumbnail((max_side, max_side), Image.LANCZOS)

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            metadata = ImageMetadata(
                filename=filename,
                modality=modality,
                crs=crs,
                bounds=bounds,
                width=width,
                height=height,
                band_count=band_count,
                dtype=dtype,
            )

            log.info("preprocessed_image",
                     filename=filename,
                     modality=modality.value,
                     bands=band_count,
                     size=f"{width}x{height}")
            return png_bytes, metadata

    finally:
        os.unlink(tmp_path)


def validate_image_pair(
    meta1: ImageMetadata,
    meta2: ImageMetadata,
    mode: str,
) -> list[str]:
    """
    Validate a pair of images for compatibility.
    Returns a list of warning strings (empty = OK).
    """
    warnings: list[str] = []

    if mode == "bitemporal":
        if meta1.modality != meta2.modality:
            warnings.append(
                f"Bi-temporal pair has mismatched modalities: "
                f"{meta1.modality} vs {meta2.modality}. "
                "Results may be less reliable."
            )
        if meta1.crs and meta2.crs and meta1.crs != meta2.crs:
            warnings.append(
                f"CRS mismatch: {meta1.crs} vs {meta2.crs}. "
                "Images should be co-registered."
            )

    elif mode == "crossmodal":
        if meta1.modality == meta2.modality:
            warnings.append(
                "Both images appear to have the same modality. "
                "Cross-modal analysis expects one optical and one SAR image."
            )
        if meta1.crs and meta2.crs and meta1.crs != meta2.crs:
            warnings.append(
                "CRS mismatch for cross-modal pair. "
                "Images should be co-registered."
            )

    return warnings


def png_to_base64(png_bytes: bytes) -> str:
    """Convert PNG bytes to base64 string for VLM API calls."""
    import base64
    return base64.b64encode(png_bytes).decode("utf-8")
