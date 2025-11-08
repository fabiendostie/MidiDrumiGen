#!/usr/bin/env python3
"""Test the complete architecture data flow: API → Redis → Celery → Result."""

import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import requests
from src.tasks.tasks import generate_pattern
from src.tasks.worker import celery_app
import redis


def test_redis_connection():
    """Test Redis connection."""
    print("=" * 70)
    print("1. Testing Redis Connection")
    print("=" * 70)

    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✓ Redis connection successful")
        return True
    except redis.ConnectionError as e:
        print(f"✗ Redis connection failed: {e}")
        return False


def test_celery_task_direct():
    """Test Celery task execution directly."""
    print("\n" + "=" * 70)
    print("2. Testing Direct Celery Task Execution")
    print("=" * 70)

    params = {
        "producer_style": "J Dilla",
        "bars": 4,
        "tempo": 95,
        "humanize": True
    }

    print(f"Queuing task with params: {params}")

    # Queue the task
    result = generate_pattern.delay(params)
    print(f"✓ Task queued with ID: {result.id}")

    # Wait for result
    print("Waiting for task to complete...")
    timeout = 30
    start_time = time.time()

    while not result.ready() and (time.time() - start_time) < timeout:
        time.sleep(0.5)
        print(".", end="", flush=True)

    print()

    if result.ready():
        if result.successful():
            task_result = result.get()
            print(f"✓ Task completed successfully")
            print(f"  Status: {task_result.get('status')}")
            print(f"  MIDI Path: {task_result.get('midi_path')}")
            return True
        else:
            print(f"✗ Task failed: {result.info}")
            return False
    else:
        print(f"✗ Task timed out after {timeout}s")
        return False


def test_api_endpoints():
    """Test API endpoints."""
    print("\n" + "=" * 70)
    print("3. Testing API Endpoints")
    print("=" * 70)

    base_url = "http://localhost:8000"

    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Root endpoint: {data.get('name')}")
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API server (is it running?)")
        return False

    # Test health endpoint
    response = requests.get(f"{base_url}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Health endpoint: {data.get('status')}")
        print(f"  Redis: {data.get('redis')}")
        return True
    else:
        print(f"✗ Health endpoint failed: {response.status_code}")
        return False


def test_celery_inspect():
    """Test Celery inspection."""
    print("\n" + "=" * 70)
    print("4. Testing Celery Worker Inspection")
    print("=" * 70)

    inspect = celery_app.control.inspect()

    # Get active workers
    active = inspect.active()
    if active:
        print(f"✓ Active workers: {list(active.keys())}")
    else:
        print("⚠ No active workers found (worker may not be running)")

    # Get registered tasks
    registered = inspect.registered()
    if registered:
        for worker, tasks in registered.items():
            print(f"✓ Worker '{worker}' has {len(tasks)} registered tasks")
            for task in tasks:
                if 'src.tasks' in task:
                    print(f"  - {task}")
    else:
        print("⚠ No registered tasks found")

    return True


def main():
    """Run all architecture tests."""
    print("\n" + "=" * 70)
    print("ARCHITECTURE DATA FLOW TEST")
    print("=" * 70)
    print()
    print("This script tests the complete architecture:")
    print("  1. Redis connection")
    print("  2. Celery task execution")
    print("  3. API endpoints")
    print("  4. Worker inspection")
    print()

    results = []

    # Run tests
    results.append(("Redis Connection", test_redis_connection()))
    results.append(("API Endpoints", test_api_endpoints()))
    results.append(("Celery Worker Inspection", test_celery_inspect()))
    results.append(("Direct Celery Task", test_celery_task_direct()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All architecture tests passed!")
        print("\nThe data flow is working correctly:")
        print("  API → Redis → Celery Worker → Result Backend → API")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        print("\nMake sure:")
        print("  1. Redis is running: docker ps | grep redis")
        print("  2. API server is running: uvicorn src.api.main:app")
        print("  3. Celery worker is running: celery -A src.tasks.worker worker")


if __name__ == "__main__":
    main()
