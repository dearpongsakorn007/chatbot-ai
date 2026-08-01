"""
สร้าง embedding จากข้อความคำถามของผู้ใช้ เพื่อเอาไปเทียบความคล้ายกับ chunks ใน Supabase
หมายเหตุ: ต้องใช้ embedding model ตัวเดียวกับที่ใช้ตอน insert ข้อมูลเข้า Supabase ตอนแรก
ไม่งั้น vector มิติไม่ตรง เทียบกันไม่ได้
ข้อมูลเดิมใน Supabase (ตาราง documents_gemini) ถูกสร้างด้วย Google Gemini embedding
(มิติ 3072) จึงต้องเรียก Gemini embedding API ไม่ใช่ OpenAI
"""
import httpx
from app.config import settings

GEMINI_EMBED_DIMENSION = 3072


async def get_embedding(text: str) -> list[float]:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.embedding_model}:embedContent"
    )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            params={"key": settings.embedding_api_key},
            json={
                "model": f"models/{settings.embedding_model}",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": GEMINI_EMBED_DIMENSION,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]["values"]
