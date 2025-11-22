"""Test suite for Phase 4: API Routes."""

import json
import sys
import time
from pathlib import Path

import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

API_BASE_URL = "http://localhost:8000"


def test_root_endpoint():
    """Test root endpoint."""
    print("\n[TEST] Root endpoint")
    response = requests.get(f"{API_BASE_URL}/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "name" in data, "Response missing 'name' field"
    print(f"[PASS] Root endpoint: {data['name']}")


def test_health_endpoint():
    """Test health check."""
    print("\n[TEST] Health endpoint")
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "status" in data, "Response missing 'status' field"
    print(f"[PASS] Health status: {data['status']}")
    if data.get("redis") == "disconnected":
        print("[WARN] Redis is not connected - Celery tasks will not work")


def test_styles_endpoint():
    """Test styles list endpoint."""
    print("\n[TEST] Styles endpoint")
    response = requests.get(f"{API_BASE_URL}/api/v1/styles")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "styles" in data, "Response missing 'styles' field"
    assert data["count"] >= 3, f"Expected at least 3 styles, got {data['count']}"
    print(f"[PASS] Styles count: {data['count']}")
    for style in data["styles"]:
        print(f"  - {style['name']}: {style['description']}")


def test_generate_endpoint():
    """Test pattern generation endpoint."""
    print("\n[TEST] Pattern generation")

    request_data = {
        "producer_style": "J Dilla",
        "bars": 4,
        "tempo": 95,
        "time_signature": [4, 4],
        "humanize": True,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.9,
    }

    response = requests.post(f"{API_BASE_URL}/api/v1/generate", json=request_data)

    assert response.status_code == 202, f"Expected 202, got {response.status_code}"
    data = response.json()
    assert "task_id" in data, "Response missing 'task_id' field"
    task_id = data["task_id"]
    print(f"[PASS] Task queued: {task_id}")

    # Poll for completion
    print("Waiting for task completion...")
    max_attempts = 30
    for i in range(max_attempts):
        time.sleep(1)
        status_response = requests.get(f"{API_BASE_URL}/api/v1/status/{task_id}")
        status_data = status_response.json()

        print(f"  Attempt {i+1}: Status = {status_data['status']}")

        if status_data["status"] == "completed":
            print("[PASS] Task completed")
            print(f"  Result: {json.dumps(status_data['result'], indent=2)}")
            return status_data

        elif status_data["status"] == "failed":
            print(f"[FAIL] Task failed: {status_data.get('error')}")
            return None

    print(f"[WARN] Task did not complete within {max_attempts} seconds")
    return None


def run_all_tests():
    """Run all API tests."""
    print("=" * 70)
    print("PHASE 4 API TEST SUITE")
    print("=" * 70)

    print("\nNOTE: Ensure API server is running:")
    print("  uvicorn src.api.main:app --reload")
    print("  celery -A src.tasks.worker worker -Q gpu_generation --loglevel=info")

    try:
        test_root_endpoint()
        test_health_endpoint()
        test_styles_endpoint()
        test_generate_endpoint()

        print("\n" + "=" * 70)
        print("[SUCCESS] All API tests passed!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n[FAIL] Test assertion failed: {e}")
        return 1
    except requests.ConnectionError:
        print("\n[FAIL] Could not connect to API server")
        print("Make sure the server is running on http://localhost:8000")
        return 1
    except Exception as e:
        print(f"\n[FAIL] Test error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
