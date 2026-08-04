import hashlib
import logging

from app.db.supabase_client import get_supabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repair-bot")
# httpx logs the complete request URL at INFO level. Embedding providers may put
# credentials in that URL, so keep transport logs above INFO in every environment.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def log_conversation(user_id: str, question: str, answer: str) -> None:
    """บันทึกสถานะแบบไม่เปิดเผยข้อความลง console และบันทึกบทสนทนาจริงลง Supabase
    (line_conversation_messages) เพื่อให้ตรวจย้อนหลังได้ว่าบอทตอบอะไรไป — จำเป็นขึ้นมาก
    หลังผ่อนเกณฑ์คัดหลักฐานใน claude_service.py เพราะไม่มีทางอื่นให้ตรวจสอบคุณภาพคำตอบจริง
    การบันทึกล้มเหลวต้องไม่ทำให้การตอบลูกค้าที่ส่งไปแล้วก่อนหน้านี้พัง จึงดักข้อผิดพลาดไว้เฉยๆ
    """
    user_ref = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]
    logger.info(
        "conversation handled user=%s question_chars=%d answer_chars=%d",
        user_ref,
        len(question),
        len(answer),
    )
    try:
        supabase = get_supabase()
        supabase.table("line_conversation_messages").insert(
            [
                {"line_user_id": user_id, "role": "user", "message_text": question},
                {"line_user_id": user_id, "role": "assistant", "message_text": answer},
            ]
        ).execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to persist conversation to Supabase: %s", exc)
