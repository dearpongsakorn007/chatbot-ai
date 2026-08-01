"""
ค้นหา chunks ที่เกี่ยวข้องกับคำถามผู้ใช้จาก Supabase pgvector
ต้องมี function/RPC ชื่อ match_chunks ตั้งไว้ใน Supabase อยู่แล้ว (ตาม pipeline OCR+chunk เดิม)
ถ้ายังไม่มี ต้องสร้าง SQL function match_chunks(query_embedding, match_count) ก่อน
"""
from app.db.supabase_client import get_supabase
from app.config import settings
from app.models.schemas import RetrievedChunk


async def retrieve_chunks(query_embedding: list[float]) -> list[RetrievedChunk]:
    supabase = get_supabase()
    result = supabase.rpc(
        "match_chunks",
        {"query_embedding": query_embedding, "match_count": settings.top_k},
    ).execute()

    rows = result.data or []
    return [
        RetrievedChunk(
            content=row.get("content", ""),
            source=row.get("source"),
            score=row.get("similarity"),
        )
        for row in rows
    ]
