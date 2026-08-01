import hashlib
import hmac
import base64

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks

from app.config import settings
from app.models.schemas import LineWebhookPayload, RetrievedChunk
from app.services.embedding_service import get_embedding
from app.services.retrieval_service import retrieve_chunks
from app.services.claude_service import ask_llm
from app.services.line_service import reply_message
from app.utils.logger import log_conversation, logger

router = APIRouter()
WARNING_TEXT = "คำเตือน: ควรให้ช่างยืนยันหน้างาน"


def _verify_signature(body: bytes, signature: str) -> bool:
    hash_ = hmac.new(settings.line_channel_secret.encode(), body, hashlib.sha256).digest()
    expected = base64.b64encode(hash_).decode()
    return hmac.compare_digest(expected, signature)


def _prepare_reply(
    answer: str,
    chunks: list[RetrievedChunk],
) -> tuple[str, list[tuple[str, str]]]:
    parts = [answer.strip()]
    images: list[tuple[str, str]] = []
    reference_chunk = next((chunk for chunk in chunks if chunk.image_url), None)

    if reference_chunk and reference_chunk.image_url:
        images.append(
            (
                reference_chunk.image_url,
                reference_chunk.preview_image_url or reference_chunk.image_url,
            )
        )
        source_name = reference_chunk.source or "คู่มือ"
        reference = reference_chunk.reference or "ไม่ระบุหน้า"
        parts.append(f"รูปอ้างอิง [1]: {source_name} หน้า {reference}")

    parts.append(WARNING_TEXT)
    return "\n\n".join(parts), images


async def _handle_message(reply_token: str, user_id: str, question: str) -> None:
    try:
        embedding = await get_embedding(question)
        chunks = await retrieve_chunks(embedding)
        answer = await ask_llm(question, chunks)
        answer, images = _prepare_reply(answer, chunks)
        await reply_message(reply_token, answer, images)
        log_conversation(user_id, question, answer)
    except Exception as e:
        logger.error(f"handle_message failed: {e}")
        await reply_message(reply_token, "ขออภัยค่ะ ระบบขัดข้อง กรุณาลองใหม่อีกครั้ง")


@router.post("/webhook")
async def line_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")

    if not _verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="invalid signature")

    payload = LineWebhookPayload.model_validate_json(body)

    for event in payload.events:
        if event.type == "message" and event.message and event.message.type == "text":
            question = event.message.text or ""
            reply_token = event.replyToken or ""
            user_id = event.source.userId if event.source else ""
            # ตอบ LINE ให้เร็วก่อน (reply token หมดอายุไว) แล้วประมวลผลจริงเบื้องหลัง
            background_tasks.add_task(_handle_message, reply_token, user_id, question)

    return {"status": "ok"}
