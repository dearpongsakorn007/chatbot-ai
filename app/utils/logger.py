import logging
from app.db.supabase_client import get_supabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repair-bot")


def log_conversation(user_id: str, question: str, answer: str) -> None:
    """บันทึกบทสนทนาลง Supabase สำหรับ debug/ปรับปรุง retrieval ทีหลัง"""
    try:
        supabase = get_supabase()
        supabase.table("conversation_logs").insert(
            {"user_id": user_id, "question": question, "answer": answer}
        ).execute()
    except Exception as e:
        logger.error(f"log_conversation failed: {e}")
