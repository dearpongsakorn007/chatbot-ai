import httpx
from app.config import settings

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
# LINE จำกัด bubble ต่อ carousel 1 อันไว้ที่ 10
MAX_CAROUSEL_IMAGES = 10


def _valid_image_pairs(
    images: list[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    seen: set[str] = set()
    valid: list[tuple[str, str]] = []
    for original_url, preview_url in images or []:
        if original_url in seen:
            continue
        if not original_url.startswith("https://") or not preview_url.startswith("https://"):
            continue
        seen.add(original_url)
        valid.append((original_url, preview_url))
        if len(valid) == MAX_CAROUSEL_IMAGES:
            break
    return valid


def _build_image_carousel(images: list[tuple[str, str]]) -> dict:
    """รูปอ้างอิงหลายหน้า (chunk เดียวครอบคลุมหลายหน้า) ให้อยู่ในกรอบเดียวกัน
    เป็น carousel เดียว ปัด/แตะเพื่อดูรูปถัดไป แทนที่จะส่งเป็นรูปแยกเรียงยาวในแชท

    รูปใน hero component ของ Flex Message ไม่มีตัวขยายเต็มจอในตัวแบบ image message ปกติ
    (แตะแล้วไม่มีอะไรเกิดขึ้น) ต้องผูก action แบบ uri ชี้ไปยังรูปต้นฉบับเอง ให้แตะแล้วเปิดดูเต็มได้
    """
    bubbles = [
        {
            "type": "bubble",
            "hero": {
                "type": "image",
                "url": original_url,
                "size": "full",
                "aspectRatio": "3:4",
                "aspectMode": "cover",
                "action": {"type": "uri", "uri": original_url},
            },
        }
        for original_url, _ in images
    ]
    return {
        "type": "flex",
        "altText": f"รูปอ้างอิง {len(bubbles)} หน้า",
        "contents": {"type": "carousel", "contents": bubbles},
    }


def _build_messages(
    text: str,
    images: list[tuple[str, str]] | None = None,
) -> list[dict]:
    messages: list[dict] = [{"type": "text", "text": text}]
    valid_images = _valid_image_pairs(images)

    if len(valid_images) == 1:
        original_url, preview_url = valid_images[0]
        messages.append(
            {
                "type": "image",
                "originalContentUrl": original_url,
                "previewImageUrl": preview_url,
            }
        )
    elif len(valid_images) > 1:
        messages.append(_build_image_carousel(valid_images))

    return messages


async def reply_message(
    reply_token: str,
    text: str,
    images: list[tuple[str, str]] | None = None,
) -> None:
    headers = {
        "Authorization": f"Bearer {settings.line_channel_access_token}",
        "Content-Type": "application/json",
    }
    messages = _build_messages(text, images)
    payload = {"replyToken": reply_token, "messages": messages}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(LINE_REPLY_URL, headers=headers, json=payload)
        if resp.is_error and len(messages) > 1:
            fallback_payload = {
                "replyToken": reply_token,
                "messages": [{"type": "text", "text": text}],
            }
            resp = await client.post(
                LINE_REPLY_URL,
                headers=headers,
                json=fallback_payload,
            )
        resp.raise_for_status()


async def push_message(user_id: str, text: str) -> None:
    """ใช้ตอน reply token หมดอายุแล้ว (เช่น กรณีประมวลผลนานเกินไป)"""
    headers = {
        "Authorization": f"Bearer {settings.line_channel_access_token}",
        "Content-Type": "application/json",
    }
    payload = {"to": user_id, "messages": [{"type": "text", "text": text}]}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(LINE_PUSH_URL, headers=headers, json=payload)
        resp.raise_for_status()
