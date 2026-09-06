"""
SatQuery AI — Real Optical + SAR Cross-Modal End-to-End Test
Executes a genuine end-to-end analysis on matching real Sentinel-1 SAR and Sentinel-2 Multispectral
GeoTIFF patch files from BigEarthNet without mocks or hardcoded responses.
"""
import io
import os
import sys
import json
import base64
import asyncio
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from datasets.bigearthnet import BigEarthNetDataset
from core.orchestrator import orchestrate_crossmodal_satquery_request


async def run_real_e2e_test():
    print("=" * 70)
    print("STARTING GENUINE END-TO-END OPTICAL + SAR CROSS-MODAL TEST")
    print("=" * 70)

    # 1. Load real test record
    split_path = os.path.join(PROJECT_ROOT, "dataset", "bigearthnet", "splits", "test.json")
    with open(split_path, "r") as f:
        recs = json.load(f)

    # Use patch S2A_MSIL2A_20170717T113321_61_51
    sample = recs[0]
    print(f"Sample Patch ID: {sample['patch_id']}")
    print(f"  Optical B02 GeoTIFF: {sample['s2_b02_path']}")
    print(f"  SAR VV GeoTIFF:      {sample['s1_vv_path']}")
    print(f"  Actual Ground Truth: {sample['labels']}")

    # 2. Encode real GeoTIFF images to base64
    opt_img = Image.open(sample["s2_b02_path"]).convert("RGB")
    buf_opt = io.BytesIO()
    opt_img.save(buf_opt, format="PNG")
    b64_optical = base64.b64encode(buf_opt.getvalue()).decode("utf-8")

    sar_img = Image.open(sample["s1_vv_path"])
    # SAR float32 normalized for display / raster transfer
    arr = np.array(sar_img, dtype=np.float32)
    norm = ((arr - (-25.0)) / 30.0 * 255.0).clip(0, 255).astype(np.uint8)
    sar_pil = Image.fromarray(norm)
    buf_sar = io.BytesIO()
    sar_pil.save(buf_sar, format="PNG")
    b64_sar = base64.b64encode(buf_sar.getvalue()).decode("utf-8")

    question = "Use the optical and SAR images together to identify built-up and water-covered regions."
    print(f"\nUser Query: '{question}'\n")

    # 3. Execute Orchestrator
    resp = await orchestrate_crossmodal_satquery_request(b64_optical, b64_sar, question)

    print("=" * 70)
    print("CROSS-MODAL EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Status:       {resp.get('status')}")
    print(f"Task:         {resp.get('task')}")
    print(f"Modalities:   {resp.get('modalities')}")
    print(f"Model:        {resp.get('model')}")
    print(f"Tools:        {resp.get('tools')}")
    print(f"Confidence:   {resp.get('confidence')}")
    print(f"Latency:      {resp.get('execution_time_ms')} ms")
    print(f"\nNatural Language Answer:\n{resp.get('answer')}")

    print("\nOptical Evidence Breakdown:")
    for ev in resp.get("optical_evidence", []):
        print(f"  - {ev}")

    print("\nSAR Evidence Breakdown:")
    for ev in resp.get("sar_evidence", []):
        print(f"  - {ev}")

    print("\nFused Evidence Breakdown:")
    for ev in resp.get("fused_evidence", []):
        print(f"  - {ev}")

    print("\nWater Analysis:")
    print(f"  {resp.get('water_analysis')}")

    print("\nBuilt-up Analysis:")
    print(f"  {resp.get('built_up_analysis')}")

    print("=" * 70)
    assert resp["status"] == "success"
    assert resp["task"] == "optical_sar_cross_modal"
    assert resp["confidence"] > 0.70
    assert len(resp["overlay"]) > 1000
    print("E2E VALIDATION: PASSED")


if __name__ == "__main__":
    import numpy as np
    asyncio.run(run_real_e2e_test())
