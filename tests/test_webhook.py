"""
เทสเบื้องต้น: health check และ webhook ปฏิเสธ signature ผิด
รันด้วย: pytest
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_webhook_rejects_invalid_signature():
    resp = client.post(
        "/webhook",
        content=b'{"destination": "x", "events": []}',
        headers={"X-Line-Signature": "invalid"},
    )
    assert resp.status_code == 401
