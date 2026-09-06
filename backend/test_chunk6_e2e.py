"""
SatQuery AI — Chunk 6 End-to-End & Microservice Integration Test Suite
Validates:
  1. Test 1: "How many red boxes are visible?"
  2. Test 2: "How many blue boxes are visible?"
  3. Test 3: "Is there a car in the image?"
  4. Test 4: "How many objects are detected?"
  5. Test 5: "Where is the car relative to the red and blue structures?"
  6. Test 6: "Describe the objects visible in the image."
  7. Failure Mode Tests (empty image, invalid Base64, empty question)
  8. Health Check Endpoint (/health & /api/satquery/health)
"""
import os
import sys
import json
import base64
import time
import asyncio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT, "hf_spaces", "grounding", "test_scene.png")


def load_test_image_b64() -> str:
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"Test image missing at '{TEST_IMAGE_PATH}'.")
    with open(TEST_IMAGE_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def run_all_e2e_tests():
    print("=" * 70)
    print("SATQUERY AI — CHUNK 6 FULL E2E ORCHESTRATOR VALIDATION")
    print("=" * 70)

    from core.orchestrator import orchestrate_satquery_request
    from routers.orchestrator import satquery_health_check

    # 0. Health Check Test
    print("\n[HEALTH CHECK TEST]")
    t0 = time.monotonic()
    health_resp = await satquery_health_check()
    duration_health = round((time.monotonic() - t0) * 1000, 2)
    print(f"Status: {health_resp['status']}")
    print(f"Services: {json.dumps(health_resp['services'], indent=2)}")
    print(f"Health check latency: {duration_health} ms")
    assert health_resp["status"] in ("healthy", "degraded"), f"Unexpected health status: {health_resp['status']}"

    b64_img = load_test_image_b64()

    tests = [
        {
            "id": 1,
            "question": "How many red boxes are visible?",
            "expected_check": lambda ans, dets: "red" in ans.lower() and len(dets) >= 1,
        },
        {
            "id": 2,
            "question": "How many blue boxes are visible?",
            "expected_check": lambda ans, dets: "blue" in ans.lower() and len(dets) >= 1,
        },
        {
            "id": 3,
            "question": "Is there a car in the image?",
            "expected_check": lambda ans, dets: "no vehicle was detected" in ans.lower(),
        },
        {
            "id": 4,
            "question": "How many objects are detected?",
            "expected_check": lambda ans, dets: "total of" in ans.lower() or "objects are detected" in ans.lower(),
        },
        {
            "id": 5,
            "question": "Where is the car relative to the red and blue structures?",
            "expected_check": lambda ans, dets: ("no car was detected" in ans.lower() or "positioned" in ans.lower() or "located" in ans.lower()) and "pixels" in ans.lower(),
        },
        {
            "id": 6,
            "question": "Describe the objects visible in the image.",
            "expected_check": lambda ans, dets: "contains" in ans.lower() and "grounded objects" in ans.lower(),
        },
    ]

    results = []

    for test in tests:
        t_start = time.monotonic()
        print(f"\n" + "-" * 60)
        print(f"TEST {test['id']}: \"{test['question']}\"")

        res = await orchestrate_satquery_request(b64_img, test['question'])
        t_end = time.monotonic()
        total_time_ms = round((t_end - t_start) * 1000, 2)

        ans = res.get("answer", "")
        dets = res.get("detections", [])

        print(f"Status: {res.get('status')}")
        print(f"Image Size: {res.get('image_size')}")
        print(f"Detections Count: {len(dets)}")
        print(f"Detections Breakdown: {[(d['label'], d['score'], d['box']) for d in dets]}")
        print(f"Answer: {ans}")
        print(f"Overlay Base64 (len): {len(res.get('overlay', ''))}")
        print(f"Orchestration Latency: {total_time_ms} ms")

        # Functional checks
        assert res.get("status") == "success", f"Test {test['id']} failed with status error: {res}"
        assert ans, f"Test {test['id']} returned empty answer"
        assert isinstance(dets, list), f"Test {test['id']} detections invalid"
        assert len(res.get("overlay", "")) > 100, f"Test {test['id']} overlay missing"
        assert test["expected_check"](ans, dets), f"Test {test['id']} answer functionally incorrect: '{ans}'"

        results.append({
            "test_id": test["id"],
            "question": test["question"],
            "status": "PASSED",
            "answer": ans,
            "detections_count": len(dets),
            "latency_ms": total_time_ms,
        })

    # Failure Mode Tests
    print("\n" + "=" * 70)
    print("FAILURE MODE TESTS")
    print("=" * 70)

    # Failure 1: Empty Image
    print("\n[FAILURE TEST 1: Empty Image Base64]")
    fail1 = await orchestrate_satquery_request("", "How many red boxes?")
    print(f"Response: {fail1}")
    assert fail1.get("status") == "error"
    assert fail1.get("error", {}).get("code") == "INVALID_IMAGE_PAYLOAD"
    print("-> PASSED: Handled cleanly with INVALID_IMAGE_PAYLOAD")

    # Failure 2: Invalid Base64 String
    print("\n[FAILURE TEST 2: Corrupt Base64 String]")
    fail2 = await orchestrate_satquery_request("not_valid_base64_!@#$", "How many red boxes?")
    print(f"Response: {fail2}")
    assert fail2.get("status") == "error"
    assert fail2.get("error", {}).get("code") in ("INVALID_IMAGE_FORMAT", "INVALID_IMAGE_PAYLOAD")
    print("-> PASSED: Handled cleanly with INVALID_IMAGE_FORMAT")

    # Failure 3: Empty Question
    print("\n[FAILURE TEST 3: Empty Question]")
    fail3 = await orchestrate_satquery_request(b64_img, "   ")
    print(f"Response: {fail3}")
    assert fail3.get("status") == "error"
    assert fail3.get("error", {}).get("code") == "EMPTY_QUESTION"
    print("-> PASSED: Handled cleanly with EMPTY_QUESTION")

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"Test {r['test_id']}: PASSED ({r['latency_ms']} ms) | Detections: {r['detections_count']}")
    print(f"\nHealth Check: PASSED ({duration_health} ms)")
    print(f"Failure Mode Tests: 3/3 PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_all_e2e_tests())
