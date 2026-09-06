"""
SatQuery AI — Multi-Band GeoTIFF Support Tests
Verifies:
  1. Single-file 4-band Sentinel-2 GeoTIFF decoding ([B02, B03, B04, B08])
  2. Single-file 2-band Sentinel-1 SAR GeoTIFF decoding ([VV, VH])
  3. Standard 3-band RGB TIFF/GeoTIFF decoding
  4. OpticalSARAnalyzer consumption of raw 4-band S2 tensor
  5. OpticalSARAnalyzer consumption of raw 2-band S1 SAR tensor
  6. Bi-temporal 4-band GeoTIFF change analysis with ChangeAnalyzer
  7. InputValidator Base64 GeoTIFF decoding with raw tensor retention
  8. JPG and PNG regression parity
"""
import io
import os
import sys
import base64
import pytest
import numpy as np
import torch
from PIL import Image
import tifffile

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessor import decode_multimodal_geotiff
from core.input_validator import get_input_validator
from core.query_interpreter import QueryIntent
from core.optical_sar_analyzer import get_optical_sar_analyzer
from core.change_analyzer import get_change_analyzer


def _make_4band_s2_bytes(width=64, height=64) -> bytes:
    """Generate in-memory 4-band Sentinel-2 uint16 GeoTIFF bytes [B02, B03, B04, B08]."""
    # Channel 0: B02 Blue, Channel 1: B03 Green, Channel 2: B04 Red, Channel 3: B08 NIR
    b02 = np.full((height, width), 400, dtype=np.uint16)
    b03 = np.full((height, width), 600, dtype=np.uint16)
    b04 = np.full((height, width), 500, dtype=np.uint16)
    b08 = np.full((height, width), 2500, dtype=np.uint16)  # High NIR
    arr = np.stack([b02, b03, b04, b08], axis=0)  # (4, H, W)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return buf.getvalue()


def _make_2band_s1_bytes(width=64, height=64) -> bytes:
    """Generate in-memory 2-band Sentinel-1 SAR float32 GeoTIFF bytes [VV, VH] in dB."""
    # Channel 0: VV (-11 dB), Channel 1: VH (-18 dB)
    vv = np.full((height, width), -11.0, dtype=np.float32)
    vh = np.full((height, width), -18.0, dtype=np.float32)
    arr = np.stack([vv, vh], axis=0)  # (2, H, W)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return buf.getvalue()


def _make_3band_rgb_tiff_bytes(width=64, height=64) -> bytes:
    """Generate in-memory 3-band RGB uint8 GeoTIFF bytes."""
    r = np.full((height, width), 200, dtype=np.uint8)
    g = np.full((height, width), 100, dtype=np.uint8)
    b = np.full((height, width), 50, dtype=np.uint8)
    arr = np.stack([r, g, b], axis=0)  # (3, H, W)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return buf.getvalue()


class TestGeoTIFFSupport:
    def test_01_4band_s2_tiff_decoding(self):
        """Verify 4-band Sentinel-2 GeoTIFF decodes to RGB image and (4, H, W) tensor."""
        raw_bytes = _make_4band_s2_bytes()
        pil_img, raw_tensor, meta = decode_multimodal_geotiff(raw_bytes, modality_hint="optical")

        assert pil_img is not None
        assert pil_img.mode == "RGB"
        assert pil_img.size == (64, 64)

        assert raw_tensor is not None
        assert isinstance(raw_tensor, torch.Tensor)
        assert raw_tensor.shape == (4, 64, 64)
        assert meta["format"] == "sentinel2_geotiff"
        assert meta["layout"] == ["B02", "B03", "B04", "B08"]
        assert meta["band_count"] == 4

        # Verify generic 4-band without S2 hint is not treated as S2
        _, generic_tensor, generic_meta = decode_multimodal_geotiff(raw_bytes, modality_hint=None)
        assert generic_tensor is None
        assert generic_meta["format"] == "generic_4band_tiff"

    def test_02_2band_s1_tiff_decoding(self):
        """Verify 2-band Sentinel-1 SAR GeoTIFF decodes to false-color RGB and (2, H, W) tensor."""
        raw_bytes = _make_2band_s1_bytes()
        pil_img, raw_tensor, meta = decode_multimodal_geotiff(raw_bytes, modality_hint="sar")

        assert pil_img is not None
        assert pil_img.mode == "RGB"
        assert pil_img.size == (64, 64)

        assert raw_tensor is not None
        assert isinstance(raw_tensor, torch.Tensor)
        assert raw_tensor.shape == (2, 64, 64)
        assert meta["format"] == "sentinel1_geotiff"
        assert meta["layout"] == ["VV", "VH"]
        assert meta["band_count"] == 2

    def test_03_3band_rgb_tiff_decoding(self):
        """Verify standard 3-band RGB TIFF produces standard RGB PIL image without channel distortion."""
        raw_bytes = _make_3band_rgb_tiff_bytes()
        pil_img, raw_tensor, meta = decode_multimodal_geotiff(raw_bytes)

        assert pil_img is not None
        assert pil_img.mode == "RGB"
        assert raw_tensor is None
        assert meta["format"] == "rgb_geotiff"
        assert meta["band_count"] == 3

    def test_04_optical_sar_analyzer_with_4band_tensor(self):
        """Verify OpticalSARAnalyzer processes raw 4-band tensor with real NDVI/NDWI calculation."""
        analyzer = get_optical_sar_analyzer()
        # Create (4, 128, 128) float32 tensor [B02, B03, B04, B08]
        b02 = torch.full((1, 128, 128), 0.15, dtype=torch.float32)
        b03 = torch.full((1, 128, 128), 0.25, dtype=torch.float32)
        b04 = torch.full((1, 128, 128), 0.18, dtype=torch.float32)
        b08 = torch.full((1, 128, 128), 0.70, dtype=torch.float32)  # Active vegetation canopy
        s2_tensor = torch.cat([b02, b03, b04, b08], dim=0)

        opt_res = analyzer.extract_optical_features(s2_tensor)
        assert opt_res["s2_tensor"].shape == (4, 128, 128)
        assert opt_res["mean_ndvi"] > 0.40  # (0.70 - 0.18) / (0.70 + 0.18) ~ 0.59
        assert opt_res["veg_pct"] > 50.0

    def test_05_optical_sar_analyzer_with_2band_tensor(self):
        """Verify OpticalSARAnalyzer processes raw 2-band SAR tensor with genuine backscatter calculation."""
        analyzer = get_optical_sar_analyzer()
        # Normalization: vv_norm = (vv_db - (-15)) / 10. For water, VV < -18 dB => norm < -0.3
        vv_norm = torch.full((1, 128, 128), -0.7, dtype=torch.float32)  # VV ~ -22 dB
        vh_norm = torch.full((1, 128, 128), -0.8, dtype=torch.float32)  # VH ~ -28 dB
        s1_tensor = torch.cat([vv_norm, vh_norm], dim=0)

        sar_res = analyzer.extract_sar_features(s1_tensor)
        assert sar_res["s1_tensor"].shape == (2, 128, 128)
        assert sar_res["water_pct"] > 90.0  # Clear specular attenuation detected

    def test_06_bitemporal_4band_tiff_processing(self):
        """Verify ChangeAnalyzer handles 4-band Sentinel-2 GeoTIFF pairs directly."""
        analyzer = get_change_analyzer()
        t1_bytes = _make_4band_s2_bytes(128, 128)
        t2_bytes = _make_4band_s2_bytes(128, 128)

        res = analyzer.analyze_change(t1_bytes, t2_bytes, "What changed between these dates?")
        assert res is not None
        assert "change_ratio" in res
        assert "severity" in res
        assert res["change_ratio"] >= 0.0

    def test_07_input_validator_tiff_base64(self):
        """Verify InputValidator validates GeoTIFF Base64 payloads and retains raw tensors."""
        validator = get_input_validator()
        opt_b64 = base64.b64encode(_make_4band_s2_bytes()).decode("utf-8")
        sar_b64 = base64.b64encode(_make_2band_s1_bytes()).decode("utf-8")

        from core.query_interpreter import get_query_interpreter
        query = "Use the optical and SAR images together to identify built-up and water-covered regions."
        intent = get_query_interpreter().interpret(query)

        inputs = {
            "image_optical": opt_b64,
            "image_sar": sar_b64,
            "modality": "optical_sar",
        }

        val_res = validator.validate_request("Analyze optical and SAR data", intent, inputs)
        assert val_res.is_valid is True
        assert "image_optical" in val_res.decoded_images
        assert "image_sar" in val_res.decoded_images
        assert val_res.decoded_tensors is not None
        assert "image_optical" in val_res.decoded_tensors
        assert val_res.decoded_tensors["image_optical"].shape == (4, 64, 64)
        assert "image_sar" in val_res.decoded_tensors
        assert val_res.decoded_tensors["image_sar"].shape == (2, 64, 64)

    def test_08_jpg_png_regression(self):
        """Ensure PNG and JPEG inputs decode cleanly via standard pathway with raw_tensor=None."""
        # Create standard PNG in memory
        img = Image.new("RGB", (64, 64), color=(120, 80, 40))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        pil_img, raw_tensor, meta = decode_multimodal_geotiff(png_bytes)
        assert pil_img.mode == "RGB"
        assert raw_tensor is None
        assert meta["format"] == "standard_raster"
