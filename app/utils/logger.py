import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repair-bot")


def log_conversation(user_id: str, question: str, answer: str) -> None:
    """บันทึกเฉพาะสถานะแบบไม่เปิดเผยข้อความ และไม่เขียนลงตารางที่ไม่มีอยู่."""
    user_ref = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]
    logger.info(
        "conversation handled user=%s question_chars=%d answer_chars=%d",
        user_ref,
        len(question),
        len(answer),
    )
