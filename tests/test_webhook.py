"""
เทสเบื้องต้น: health check และ webhook ปฏิเสธ signature ผิด
รันด้วย: pytest
"""
from fastapi.testclient import TestClient
from app.main import app
from app.models.schemas import RetrievedChunk
from app.routers.webhook import (
    _is_conversation_opener,
    _model_followup_response,
    _opening_response,
    _prepare_reply,
)

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
        "อยู่มั้ยครับ",
        "อยู่หรือเปล่า?",
        "เริ่มใช้งาน",
        "ช่วยอะไรได้บ้าง",
        "ขอสอบถามหน่อย",
    ]
    assert all(_is_conversation_opener(text) for text in openers)


def test_real_question_is_not_treated_as_opener():
    assert not _is_conversation_opener("สวัสดีครับ เครื่อง SK200-8 ขึ้น Error O908")


def test_opening_response_is_a_broad_question():
    response = _opening_response("สวัสดีครับ")
    assert response == "สวัสดีครับ บริษัท TIS คุณลูกค้าสนใจสอบถามรถรุ่นไหนครับ"


def test_model_only_message_gets_topic_followup():
    response = _model_followup_response("รุ่น sk 200-8 ครับ")
    assert response is not None
    assert response.startswith("รุ่น SK200-8 นะครับ")
    assert "วิธีใช้งาน" in response
    assert "การบำรุงรักษา" in response
    assert "Error Code" in response


def test_model_with_problem_is_not_intercepted():
    assert _model_followup_response("SK200-8 เครื่องร้อนและมีเสียงหอน") is None
    assert _model_followup_response("Error P0217") is None


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
    assert text.startswith("สวัสดีครับช่างเต้ TIS ครับ\n\nคำตอบครับ")
    assert "รูปอ้างอิง: หน้า 21-33" in text
    assert "[1]" not in text
    assert "รูปอ้าง: หน้า 21-33ครับ" not in text
    assert text.endswith("คำเตือน: ควรให้ช่างยืนยันหน้างานครับ")
    assert "22-3" not in text
    assert images == [
        ("https://example.com/primary.jpg", "https://example.com/primary.jpg")
    ]


def test_prepare_reply_attaches_every_page_image_the_primary_chunk_covers():
    chunk = RetrievedChunk(
        content="primary",
        reference="17-39 to 17-41",
        image_url="https://example.com/p1071.jpg",
        preview_image_url="https://example.com/preview1071.jpg",
        page_image_urls=[
            "https://example.com/p1071.jpg",
            "https://example.com/p1072.jpg",
            "https://example.com/p1073.jpg",
        ],
    )
    text, images = _prepare_reply("คำตอบ", [chunk])
    assert "รูปอ้างอิง: หน้า 17-39 to 17-41" in text
    assert images == [
        ("https://example.com/p1071.jpg", "https://example.com/preview1071.jpg"),
        ("https://example.com/p1072.jpg", "https://example.com/p1072.jpg"),
        ("https://example.com/p1073.jpg", "https://example.com/p1073.jpg"),
    ]


def test_prepare_reply_does_not_borrow_image_from_secondary_result():
    chunks = [
        RetrievedChunk(content="primary", reference="21-33"),
        RetrievedChunk(
            content="secondary",
            reference="22-3",
            image_url="https://example.com/secondary.jpg",
        ),
    ]
    text, images = _prepare_reply("คำตอบ", chunks)
    assert "รูปอ้างอิง" not in text
    assert images == []


def test_prepare_reply_adds_polite_ending_to_every_answer_line():
    text, _ = _prepare_reply("สาเหตุ: ปั๊มแรงดันต่ำ\nวิธีแก้: ตรวจปั๊มครับ", [])
    assert "สาเหตุ: ปั๊มแรงดันต่ำครับ" in text
    assert "วิธีแก้: ตรวจปั๊มครับครับ" not in text
