"""
ค้นหา chunks ที่เกี่ยวข้องกับคำถามผู้ใช้จาก Supabase pgvector
ใช้ function/RPC ชื่อ match_documents ซึ่งรองรับ embeddings ชุดปัจจุบัน
"""
from app.db.supabase_client import get_supabase
from app.config import settings
from app.models.schemas import RetrievedChunk


async def retrieve_chunks(query_embedding: list[float]) -> list[RetrievedChunk]:
    supabase = get_supabase()
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": settings.top_k,
            "filter": {},
        },
    ).execute()

    rows = result.data or []
    chunks = []
    for row in rows:
        metadata = row.get("metadata") or {}
        chunks.append(
            RetrievedChunk(
                content=row.get("content", ""),
                source=row.get("source") or metadata.get("source"),
                score=row.get("similarity"),
            )
        )
    return chunks
