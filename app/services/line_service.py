import httpx
from app.config import settings

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
MAX_REPLY_IMAGES = 4


def _build_messages(
    text: str,
    images: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"type": "text", "text": text}]
    seen: set[str] = set()

    for original_url, preview_url in images or []:
        if len(messages) > MAX_REPLY_IMAGES:
            break
        if original_url in seen:
            continue
        if not original_url.startswith("https://") or not preview_url.startswith("https://"):
            continue
        seen.add(original_url)
        messages.append(
            {
                "type": "image",
                "originalContentUrl": original_url,
                "previewImageUrl": preview_url,
            }
        )

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
