"""เทส _build_messages: ส่งรูปอ้างอิงเป็น image message แยกทีละรูป (แตะดูเต็มจอในแอพ LINE ได้)"""
from app.services.line_service import MAX_REPLY_IMAGES, _build_messages, _valid_image_pairs


def test_build_messages_sends_plain_image_for_single_reference():
    messages = _build_messages(
        "คำตอบ",
        [("https://example.com/p1.jpg", "https://example.com/p1.jpg")],
    )
    assert len(messages) == 2
    assert messages[1] == {
        "type": "image",
        "originalContentUrl": "https://example.com/p1.jpg",
        "previewImageUrl": "https://example.com/p1.jpg",
    }


def test_build_messages_sends_each_page_image_as_its_own_native_image_message():
    images = [
        ("https://example.com/p1.jpg", "https://example.com/p1.jpg"),
        ("https://example.com/p2.jpg", "https://example.com/p2.jpg"),
        ("https://example.com/p3.jpg", "https://example.com/p3.jpg"),
    ]
    messages = _build_messages("คำตอบ", images)
    assert len(messages) == 4
    assert [m["type"] for m in messages] == ["text", "image", "image", "image"]
    assert messages[2] == {
        "type": "image",
        "originalContentUrl": "https://example.com/p2.jpg",
        "previewImageUrl": "https://example.com/p2.jpg",
    }


def test_build_messages_deduplicates_and_rejects_non_https_urls():
    images = [
        ("https://example.com/p1.jpg", "https://example.com/p1.jpg"),
        ("https://example.com/p1.jpg", "https://example.com/p1.jpg"),
        ("http://insecure.example.com/p2.jpg", "http://insecure.example.com/p2.jpg"),
    ]
    messages = _build_messages("คำตอบ", images)
    assert len(messages) == 2
    assert messages[1]["type"] == "image"


def test_build_messages_sends_only_text_when_no_valid_images():
    assert _build_messages("คำตอบ", []) == [{"type": "text", "text": "คำตอบ"}]


def test_valid_image_pairs_caps_at_max_reply_images():
    images = [(f"https://example.com/p{i}.jpg", f"https://example.com/p{i}.jpg") for i in range(15)]
    capped = _valid_image_pairs(images)
    assert len(capped) == MAX_REPLY_IMAGES
