"""
เทสเบื้องต้น: health check และ webhook ปฏิเสธ signature ผิด
รันด้วย: pytest
"""
from fastapi.testclient import TestClient
from app.main import app
from app.models.schemas import RetrievedChunk
from app.routers.webhook import _is_conversation_opener, _opening_response, _prepare_reply

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


def test_conversation_openers_cover_common_variants():
    openers = [
        "สวัสดีครับ",
        "หวัดดีค่ะ 👋",
        "HELLO",
        "Hi bot!",
        "มีใครอยู่ไหม?",
        "เริ่มใช้งาน",
        "ช่วยอะไรได้บ้าง",
        "ขอสอบถามหน่อย",
    ]
    assert all(_is_conversation_opener(text) for text in openers)


def test_real_question_is_not_treated_as_opener():
    assert not _is_conversation_opener("สวัสดีครับ เครื่อง SK200-8 ขึ้น Error O908")


def test_opening_response_is_a_broad_question():
    response = _opening_response("สวัสดีครับ")
    assert "?" in response
    assert any(word in response for word in ("รุ่นเครื่อง", "อาการ", "วิธีใช้งาน"))


def test_prepare_reply_locks_the_first_ranked_image_reference():
    chunks = [
        RetrievedChunk(
            content="primary",
            reference="21-33",
            image_url="https://example.com/primary.jpg",
        ),
        RetrievedChunk(
            content="secondary",
            reference="22-3",
            image_url="https://example.com/secondary.jpg",
        ),
    ]
    text, images = _prepare_reply("คำตอบ", chunks)
    assert text.startswith("สวัสดีครับช่างเต้ครับ\n\nคำตอบ")
    assert "หน้า 21-33" in text
    assert "22-3" not in text
    assert images == [
        ("https://example.com/primary.jpg", "https://example.com/primary.jpg")
    ]
