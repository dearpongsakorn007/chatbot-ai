import hashlib
import hmac
import base64
import re

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks

from app.config import settings
from app.models.schemas import LineWebhookPayload, RetrievedChunk
from app.services.conversation_state_service import (
    clear_pending_clarification,
    get_pending_clarification,
    save_pending_clarification,
)
from app.services.embedding_service import get_embedding
from app.services.retrieval_service import lookup_error_code, retrieve_chunks
from app.services.claude_service import (
    ask_clarifying_question,
    ask_llm,
    extract_error_code,
    get_ambiguity_clarification,
    infer_content_type_filter,
    infer_search_category_hint,
    is_insufficient_data_answer,
    rerank_chunks,
    rewrite_search_queries,
)
from app.services.line_service import reply_message
from app.utils.logger import log_conversation, logger

router = APIRouter()
ANSWER_GREETING = "สวัสดีครับช่างเต้ TIS ครับ"
WARNING_TEXT = "คำเตือน: ควรให้ช่างยืนยันหน้างานครับ"
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
    "อยู่มั้ย",
    "อยู่มั้ยครับ",
    "อยู่มั้ยค่ะ",
    "อยู่มั้ยคับ",
    "อยู่หรือเปล่า",
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
OPENING_RESPONSE = "สวัสดีครับ บริษัท TIS คุณลูกค้าสนใจสอบถามรถรุ่นไหนครับ"
# กันบอทถามกลับวนไม่จบเมื่อข้อมูลไม่มีอยู่ในคู่มือจริงๆ (เจอจริง: ถามเรื่องเกรดน้ำมันไฮดรอลิก
# ที่ไม่มีข้อมูลในคู่มือเล่มนี้เลย บอทเลยถามกลับซ้ำไปเรื่อยๆ ไม่เคยได้คำตอบสักที)
MAX_CLARIFICATION_ROUNDS = 2
GIVE_UP_ANSWER = (
    "ไม่พบข้อมูลนี้ในคู่มือที่มีอยู่ กรุณาสอบถามช่างอาวุโสหรือระบุ Error Code ที่ขึ้นจอเพิ่มเติมครับ"
)
MODEL_ONLY_PATTERN = re.compile(
    r"^\s*(?:(?:รถ\s*)?รุ่น\s*)?"
    r"(?:(?:kobelco|komatsu|hitachi|volvo|cat(?:erpillar)?)\s+)?"
    r"([a-z]{2,10}\s*[- ]?\s*\d{2,4}(?:\s*[- ]\s*\d{1,3})?[a-z0-9-]*)"
    r"\s*(?:ครับ|ค่ะ|คับ)?\s*$",
    re.IGNORECASE,
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
    return OPENING_RESPONSE


def _model_followup_response(text: str) -> str | None:
    match = MODEL_ONLY_PATTERN.fullmatch(text)
    if not match:
        return None
    model = re.sub(r"\s+", "", match.group(1)).upper()
    return (
        f"รุ่น {model} นะครับ ต้องการสอบถามเรื่องใดครับ เช่น วิธีใช้งาน "
        "การบำรุงรักษา อาการเสีย หรือ Error Code อะไรครับ?"
    )


def _ensure_polite_ending(text: str) -> str:
    """เติมคำลงท้ายให้ทุกบรรทัดของคำตอบ โดยไม่เติมซ้ำเมื่อมีอยู่แล้ว."""
    lines = []
    for raw_line in text.strip().splitlines():
        line = raw_line.rstrip()
        if line and not re.search(r"ครับ[.!?…]*$", line):
            line = f"{line}ครับ"
        lines.append(line)
    return "\n".join(lines)


def _prepare_reply(
    answer: str,
    chunks: list[RetrievedChunk],
) -> tuple[str, list[tuple[str, str]]]:
    parts = [ANSWER_GREETING, _ensure_polite_ending(answer)]
    images: list[tuple[str, str]] = []
    # The reranker's first result is the only permitted page/image reference.
    reference_chunk = chunks[0] if chunks and chunks[0].image_url else None

    if reference_chunk and reference_chunk.image_url:
        # chunk เดียวอาจครอบคลุมหลายหน้า (page_image_urls) แนบรูปทุกหน้าที่คำตอบอ้างอิงถึงจริง
        # ไม่ใช่แค่หน้าแรก ไม่งั้นช่างจะพลาดไดอะแกรม/ตารางที่อยู่หน้าอื่นในเนื้อหาเดียวกัน
        page_images = reference_chunk.page_image_urls or [reference_chunk.image_url]
        for url in page_images:
            if url == reference_chunk.image_url:
                images.append((url, reference_chunk.preview_image_url or url))
            else:
                images.append((url, url))
        reference = reference_chunk.reference or "ไม่ระบุหน้า"
        parts.append(f"รูปอ้างอิง: หน้า {reference}")

    parts.append(WARNING_TEXT)
    return "\n\n".join(parts), images


async def _clarify_or_give_up(user_id: str, question: str, rounds: int) -> str:
    """ถามกลับต่อได้ถ้ายังไม่ครบเพดานรอบ ไม่งั้นเลิกถามแล้วบอกตรงๆ ว่าไม่พบข้อมูล

    กันกรณีข้อมูลไม่มีอยู่ในคู่มือจริงๆ (เช่น เกรดน้ำมันไฮดรอลิก) ทำให้ถามกลับวนไม่จบ
    """
    if rounds >= MAX_CLARIFICATION_ROUNDS:
        clear_pending_clarification(user_id)
        return GIVE_UP_ANSWER
    answer = await ask_clarifying_question(question)
    save_pending_clarification(user_id, question, rounds + 1)
    return answer


async def _handle_message(reply_token: str, user_id: str, question: str) -> None:
    try:
        if _is_conversation_opener(question):
            clear_pending_clarification(user_id)
            answer = _opening_response(question)
            await reply_message(reply_token, answer)
            log_conversation(user_id, question, answer)
            return

        model_followup = _model_followup_response(question)
        if model_followup:
            clear_pending_clarification(user_id)
            await reply_message(reply_token, model_followup)
            log_conversation(user_id, question, model_followup)
            return

        # ถ้าบอทเพิ่งถามกลับไปหาผู้ใช้คนนี้และยังไม่หมดอายุ ให้ผูกคำตอบสั้นๆ นี้เข้ากับคำถามเดิม
        # ก่อนค้นหา ไม่งั้นข้อความสั้นๆ (เช่น "ระบบบิดครับ") จะถูกค้นหาแบบไม่มีบริบทแล้วตอบผิดเรื่อง
        pending = get_pending_clarification(user_id)
        clarification_rounds = 0
        if pending:
            pending_question, clarification_rounds = pending
            clear_pending_clarification(user_id)
            question = f"{pending_question} {question}"

        clarification = get_ambiguity_clarification(question)
        if clarification:
            save_pending_clarification(user_id, question, clarification_rounds + 1)
            answer, images = _prepare_reply(clarification, [])
            await reply_message(reply_token, answer, images)
            log_conversation(user_id, question, answer)
            return

        error_code = extract_error_code(question)
        if error_code:
            verified_chunk = await lookup_error_code(error_code)
            if verified_chunk:
                answer = await ask_llm(question, [verified_chunk])
                reply_chunks = [verified_chunk]
                if is_insufficient_data_answer(answer):
                    # จะตอบว่าไม่พบข้อมูลอยู่แล้ว ไม่มีอะไรจะเสีย เปลี่ยนเป็นถามกลับเจาะจงแทน
                    # (หรือเลิกถามถ้าครบเพดานแล้ว) และไม่แนบรูป/เลขหน้าของ chunk นี้
                    # ไม่งั้นคำตอบจะขัดแย้งกันเอง
                    answer = await _clarify_or_give_up(
                        user_id, question, clarification_rounds
                    )
                    reply_chunks = []
                answer, images = _prepare_reply(answer, reply_chunks)
                await reply_message(reply_token, answer, images)
                log_conversation(user_id, question, answer)
                return

        search_queries = await rewrite_search_queries(question)
        embedding_text = question
        if search_queries:
            embedding_text += f"\nTechnical manual terms: {search_queries[0]}"
        category_hint = infer_search_category_hint(question)
        if category_hint:
            embedding_text += f"\nManual category: {category_hint}"
        embedding = await get_embedding(embedding_text)
        content_type_filter = infer_content_type_filter(question)
        chunks = await retrieve_chunks(embedding, search_queries, content_type_filter)
        reranked = await rerank_chunks(question, chunks, search_queries)
        chunks = reranked.chunks
        if chunks:
            answer = await ask_llm(question, chunks)
            if is_insufficient_data_answer(answer):
                # จะตอบว่าไม่พบข้อมูลอยู่แล้ว ไม่มีอะไรจะเสีย เปลี่ยนเป็นถามกลับเจาะจงแทน
                # (หรือเลิกถามถ้าครบเพดานแล้ว) และไม่แนบรูปอ้างอิงของ chunk นี้
                answer = await _clarify_or_give_up(user_id, question, clarification_rounds)
                chunks = []
        else:
            answer = await _clarify_or_give_up(user_id, question, clarification_rounds)
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
