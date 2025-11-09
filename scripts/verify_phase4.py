"""Quick verification script for Phase 4."""

import requests
import json
import time
import sys

# Fix Unicode output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

API_BASE = "http://localhost:8000"

print("="*70)
print("PHASE 4 VERIFICATION")
print("="*70)

# Test 1: API is running
print("\n[1/5] Testing API server...")
try:
    response = requests.get(f"{API_BASE}/")
    assert response.status_code == 200
    print("✅ API server is running")
except Exception as e:
    print(f"❌ API server failed: {e}")
    exit(1)

# Test 2: Health check
print("\n[2/5] Testing health endpoint...")
try:
    response = requests.get(f"{API_BASE}/health")
    data = response.json()
    print(f"✅ Health: {data['status']}")
    print(f"   Redis: {data['redis']}")
    if data['redis'] != 'connected':
        print("⚠️  Warning: Redis not connected. Celery tasks won't work.")
except Exception as e:
    print(f"❌ Health check failed: {e}")

# Test 3: Styles endpoint
print("\n[3/5] Testing styles endpoint...")
try:
    response = requests.get(f"{API_BASE}/api/v1/styles")
    data = response.json()
    print(f"✅ Styles endpoint OK ({data['count']} styles available)")
    for style in data['styles']:
        print(f"   - {style['name']}")
except Exception as e:
    print(f"❌ Styles endpoint failed: {e}")

# Test 4: Documentation endpoints
print("\n[4/5] Testing documentation endpoints...")
try:
    response = requests.get(f"{API_BASE}/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()
    print("✅ Swagger UI accessible at http://localhost:8000/docs")

    response = requests.get(f"{API_BASE}/redoc")
    assert response.status_code == 200
    assert "redoc" in response.text.lower()
    print("✅ ReDoc accessible at http://localhost:8000/redoc")
except Exception as e:
    print(f"❌ Documentation endpoints failed: {e}")

# Test 5: Generate endpoint (with Redis check)
print("\n[5/5] Testing pattern generation...")
try:
    # Check Redis first
    health = requests.get(f"{API_BASE}/health").json()
    if health['redis'] != 'connected':
        print("⚠️  Skipping generation test (Redis not connected)")
    else:
        request_data = {
            "producer_style": "J Dilla",
            "bars": 2,  # Small for quick test
            "tempo": 95,
            "humanize": True
        }

        response = requests.post(
            f"{API_BASE}/api/v1/generate",
            json=request_data
        )

        assert response.status_code == 202
        data = response.json()
        task_id = data['task_id']
        print(f"✅ Task queued: {task_id}")

        # Check status
        print("   Checking task status...")
        for i in range(10):
            time.sleep(1)
            status = requests.get(f"{API_BASE}/api/v1/status/{task_id}").json()
            if status['status'] == 'completed':
                print(f"✅ Task completed!")
                print(f"   MIDI file: {status['result']['midi_file']}")
                break
            elif status['status'] == 'failed':
                print(f"❌ Task failed: {status.get('error')}")
                break
            print(f"   Status: {status['status']} (progress: {status.get('progress', 0)}%)")
        else:
            print("⚠️  Task still running after 10 seconds")

except Exception as e:
    print(f"❌ Generation test failed: {e}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("\nYou can now access:")
print("  • Swagger UI: http://localhost:8000/docs")
print("  • ReDoc: http://localhost:8000/redoc")
print("  • API Root: http://localhost:8000/")
