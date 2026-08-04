"""จำ context สั้นๆ ตอนบอทถามกลับ (clarification) แล้วผูกคำตอบสั้นๆ ของลูกค้ากลับเข้ากับคำถามเดิม

เจอจริงจากบทสนทนา: บอทถามกลับว่าน้ำมันไฮดรอลิกใช้กับระบบไหน ลูกค้าตอบสั้นๆ ว่า "ระบบบิดครับ"
ระบบเดิม (stateless ทั้งหมด) เอาข้อความนี้ไปค้นหาเดี่ยวๆ โดยไม่มีบริบทคำถามเดิม จับคำว่า "บิด"
ไปแมตช์กับตารางแรงบิดขันน็อตแทน ตอบผิดเรื่องไปเลย ใช้ตาราง line_conversation_state ที่มีอยู่แล้ว
แต่ไม่เคยถูกใช้งาน (ออกแบบไว้เป็น slot-filling state ต่อผู้ใช้ 1 แถว หมดอายุ 30 นาทีอยู่แล้ว)
"""
import logging
from datetime import datetime, timedelta, timezone

from app.db.supabase_client import get_supabase

logger = logging.getLogger("repair-bot")

_STATE_TABLE = "line_conversation_state"
# ตาราง line_conversation_state มี check constraint จำกัดค่า status ไว้แค่ "collecting" กับ "ready"
# เท่านั้น (เช็คจริงจากฐานข้อมูล) ใช้ "collecting" แทนสถานะ "กำลังรอคำตอบชี้แจง"
_AWAITING_STATUS = "collecting"
_EXPIRY_MINUTES = 30


def get_pending_clarification(user_id: str) -> str | None:
    """คืนคำถามเดิมที่บอทเพิ่งถามกลับไปหาผู้ใช้คนนี้ ถ้ายังไม่หมดอายุ ไม่งั้นคืน None"""
    try:
        supabase = get_supabase()
        now_iso = datetime.now(timezone.utc).isoformat()
        result = (
            supabase.table(_STATE_TABLE)
            .select("question_detail,expires_at")
            .eq("line_user_id", user_id)
            .eq("status", _AWAITING_STATUS)
            .gt("expires_at", now_iso)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read conversation state: %s", exc)
        return None
    rows = result.data or []
    return rows[0].get("question_detail") if rows else None


def save_pending_clarification(user_id: str, original_question: str) -> None:
    """บันทึกว่ากำลังรอคำตอบชี้แจงคำถามนี้อยู่ เก็บแค่แถวเดียวต่อผู้ใช้ (upsert ทับของเดิม)"""
    try:
        supabase = get_supabase()
        expires_at = (
            datetime.now(timezone.utc) + timedelta(minutes=_EXPIRY_MINUTES)
        ).isoformat()
        supabase.table(_STATE_TABLE).upsert(
            {
                "line_user_id": user_id,
                "question_detail": original_question,
                "status": _AWAITING_STATUS,
                "context": {},
                "expires_at": expires_at,
            },
            on_conflict="line_user_id",
        ).execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to save conversation state: %s", exc)


def clear_pending_clarification(user_id: str) -> None:
    """ล้าง state หลังใช้ไปแล้ว หรือตอนผู้ใช้เริ่มบทสนทนาใหม่ (ทักทาย/ระบุแค่รุ่นรถ)"""
    try:
        supabase = get_supabase()
        supabase.table(_STATE_TABLE).delete().eq("line_user_id", user_id).execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to clear conversation state: %s", exc)
