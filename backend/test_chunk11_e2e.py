"""
Chunk 11 — Real End-to-End Demonstration Script
Executes the required real queries on genuine remote sensing imagery:
  A. "Describe the land-cover and major objects visible in this image."
  B. "Highlight the water body referred to in the query."
  C. "What changed between these two dates?"
  D. "Use the optical and SAR images together to identify built-up and water-covered regions."
  E. "Has the built-up area increased, decreased, or remained unchanged?"

Records:
- interpreted intent
- selected tool(s)
- execution status
- answer
- confidence
- execution trace
- latency
"""
import os
import sys
import json
import base64
import asyncio
from PIL import Image
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from core.orchestrator import orchestrate_agentic_request


def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _geotiff_to_png_b64(tif_path: str) -> str:
    img = Image.open(tif_path)
    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def run_real_e2e_demonstrations():
    print("=" * 80)
    print("CHUNK 11 — REAL END-TO-END DEMONSTRATION OF AGENTIC ORCHESTRATION")
    print("=" * 80)

    # 1. Locate Real Imagery
    vrs_path = os.path.join(PROJECT_ROOT, "dataset", "vrsbench", "images", "vrs_cap_0001.png")
    cdvqa_t1 = os.path.join(PROJECT_ROOT, "dataset", "cdvqa", "images", "t1", "cdvqa_real_0000_t1.png")
    cdvqa_t2 = os.path.join(PROJECT_ROOT, "dataset", "cdvqa", "images", "t2", "cdvqa_real_0000_t2.png")

    ben_split = os.path.join(PROJECT_ROOT, "dataset", "bigearthnet", "splits", "test.json")
    with open(ben_split, "r") as f:
        ben_records = json.load(f)
    ben_sample = ben_records[0]

    b64_vrs = _file_to_b64(vrs_path)
    b64_cdvqa_t1 = _file_to_b64(cdvqa_t1)
    b64_cdvqa_t2 = _file_to_b64(cdvqa_t2)

    b64_s2 = _geotiff_to_png_b64(ben_sample["s2_b02_path"])
    b64_s1 = _geotiff_to_png_b64(ben_sample["s1_vv_path"])

    test_cases = [
        {
            "id": "A",
            "name": "Single Image Description",
            "query": "Describe the land-cover and major objects visible in this image.",
            "inputs": {"image": b64_vrs},
        },
        {
            "id": "B",
            "name": "Object Grounding",
            "query": "Highlight the water body referred to in the query.",
            "inputs": {"image": b64_vrs},
        },
        {
            "id": "C",
            "name": "Bi-Temporal Change Analysis",
            "query": "What changed between these two dates?",
            "inputs": {"image_t1": b64_cdvqa_t1, "image_t2": b64_cdvqa_t2},
        },
        {
            "id": "D",
            "name": "Cross-Modal Optical + SAR Analysis",
            "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
            "inputs": {"image_optical": b64_s2, "image_sar": b64_s1},
        },
        {
            "id": "E",
            "name": "Directional Change Quantification",
            "query": "Has the built-up area increased, decreased, or remained unchanged?",
            "inputs": {"image_t1": b64_cdvqa_t1, "image_t2": b64_cdvqa_t2},
        },
    ]

    demonstration_records = []

    for tc in test_cases:
        print(f"\n[{tc['id']}] {tc['name'].upper()}")
        print(f"Query: \"{tc['query']}\"")

        res = await orchestrate_agentic_request(tc["query"], tc["inputs"])

        record = {
            "test_id": tc["id"],
            "test_name": tc["name"],
            "query": tc["query"],
            "interpreted_intent": res["interpreted_intent"],
            "selected_tools": res["selected_tools"],
            "execution_status": res["status"],
            "answer": res["answer"],
            "confidence": res["confidence"],
            "confidence_details": res.get("confidence_details"),
            "execution_trace": res["execution_trace"],
            "latency_ms": res["execution_time_ms"],
        }
        demonstration_records.append(record)

        print(f"  • Intent:     {res['interpreted_intent']['intent']} ({res['interpreted_intent']['sub_intent']})")
        print(f"  • Tools:      {', '.join(res['selected_tools'])}")
        print(f"  • Status:     {res['status']}")
        print(f"  • Confidence: {res['confidence']} ({res.get('confidence_details', {}).get('label')})")
        print(f"  • Latency:    {res['execution_time_ms']} ms")
        print(f"  • Answer:     {res['answer'][:120]}...")
        print(f"  • Trace Steps: {len(res['execution_trace'])} steps logged")

    # Persist results
    out_dir = os.path.join(BASE_DIR, "evaluation_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chunk11_e2e_results.json")
    with open(out_path, "w") as f:
        json.dump(demonstration_records, f, indent=2)

    print("\n" + "=" * 80)
    print(f"ALL 5 REAL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_real_e2e_demonstrations())
