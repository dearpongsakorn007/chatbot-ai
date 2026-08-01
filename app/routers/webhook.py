import hashlib
import hmac
import base64
import re

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
CONVERSATION_OPENERS = {
    "สวัสดี",
    "สวัสดีครับ",
    "สวัสดีค่ะ",
    "สวัสดีคับ",
    "หวัดดี",
    "หวัดดีครับ",
    "หวัดดีค่ะ",
    "ดีครับ",
    "ดีค่ะ",
    "ทักทาย",
    "อรุณสวัสดิ์",
    "สวัสดีตอนเช้า",
    "สวัสดีตอนบ่าย",
    "สวัสดีตอนเย็น",
    "hello",
    "hellobot",
    "hi",
    "hibot",
    "hey",
    "heybot",
    "goodmorning",
    "goodafternoon",
    "goodevening",
    "มีใครอยู่ไหม",
    "อยู่ไหม",
    "เริ่มต้น",
    "เริ่มใช้งาน",
    "เริ่มสนทนา",
    "สอบถาม",
    "สอบถามหน่อย",
    "ขอสอบถามหน่อย",
    "ขอถามหน่อย",
    "ช่วยอะไรได้บ้าง",
    "ทำอะไรได้บ้าง",
    "แนะนำหน่อย",
}
OPENING_RESPONSES = (
    "สวัสดีครับ ต้องการสอบถามเรื่องใดครับ เช่น วิธีใช้งาน การบำรุงรักษา อาการเสีย หรือ Error Code?",
    "ยินดีช่วยครับ เครื่องมีอาการอย่างไร หรือมี Error Code อะไรขึ้นครับ? กรุณาระบุรุ่นเครื่องด้วยครับ",
    "ต้องการให้ช่วยตรวจสอบเรื่องใดครับ? บอกรุ่นเครื่อง อาการที่พบ หรือ Error Code ได้เลยครับ",
    "สอบถามได้เลยครับ ต้องการข้อมูลด้านการใช้งาน การซ่อมบำรุง หรือการแก้ Error Code เรื่องใดครับ?",
)


def _verify_signature(body: bytes, signature: str) -> bool:
    hash_ = hmac.new(settings.line_channel_secret.encode(), body, hashlib.sha256).digest()
    expected = base64.b64encode(hash_).decode()
    return hmac.compare_digest(expected, signature)


def _normalize_message(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold(), flags=re.UNICODE)


def _is_conversation_opener(text: str) -> bool:
    return _normalize_message(text) in {
        _normalize_message(opener) for opener in CONVERSATION_OPENERS
    }


def _opening_response(text: str) -> str:
    normalized = _normalize_message(text)
    index = sum(map(ord, normalized)) % len(OPENING_RESPONSES)
    return OPENING_RESPONSES[index]


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
        reference = reference_chunk.reference or "ไม่ระบุหน้า"
        parts.append(f"รูปอ้างอิง [1]: หน้า {reference}")

    parts.append(WARNING_TEXT)
    return "\n\n".join(parts), images


async def _handle_message(reply_token: str, user_id: str, question: str) -> None:
    try:
        if _is_conversation_opener(question):
            answer = _opening_response(question)
            await reply_message(reply_token, answer)
            log_conversation(user_id, question, answer)
            return

        embedding = await get_embedding(question)
        chunks = await retrieve_chunks(embedding, question)
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
