"""เทส _build_messages: รูปเดียวส่งแบบ image message เดิม หลายรูปให้รวมเป็น carousel เดียว"""
from app.services.line_service import _build_image_carousel, _build_messages


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


def test_build_messages_groups_multiple_page_images_into_one_carousel():
    images = [
        ("https://example.com/p1.jpg", "https://example.com/p1.jpg"),
        ("https://example.com/p2.jpg", "https://example.com/p2.jpg"),
        ("https://example.com/p3.jpg", "https://example.com/p3.jpg"),
    ]
    messages = _build_messages("คำตอบ", images)
    assert len(messages) == 2
    flex = messages[1]
    assert flex["type"] == "flex"
    assert flex["contents"]["type"] == "carousel"
    assert len(flex["contents"]["contents"]) == 3
    assert flex["contents"]["contents"][1]["hero"]["url"] == "https://example.com/p2.jpg"


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


def test_build_image_carousel_caps_at_ten_bubbles():
    images = [(f"https://example.com/p{i}.jpg", f"https://example.com/p{i}.jpg") for i in range(15)]
    from app.services.line_service import _valid_image_pairs

    capped = _valid_image_pairs(images)
    carousel = _build_image_carousel(capped)
    assert len(carousel["contents"]["contents"]) == 10
