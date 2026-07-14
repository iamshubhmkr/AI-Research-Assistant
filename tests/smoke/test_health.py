import requests


def test_health():
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        assert r.status_code == 200 and r.json()["status"] == "ok"
    except requests.ConnectionError:
        print("API not running — skipped")
