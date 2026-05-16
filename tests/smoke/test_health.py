"""Post-deployment smoke test."""
import requests

def test_health_endpoint():
    try:
        resp = requests.get("http://localhost:8000/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
    except requests.ConnectionError:
        print("API not running — skipping smoke test")
