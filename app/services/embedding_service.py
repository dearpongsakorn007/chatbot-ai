"""
สร้าง embedding จากข้อความคำถามของผู้ใช้ เพื่อเอาไปเทียบความคล้ายกับ chunks ใน Supabase
หมายเหตุ: ต้องใช้ embedding model ตัวเดียวกับที่ใช้ตอน insert ข้อมูลเข้า Supabase ตอนแรก
ไม่งั้น vector มิติไม่ตรง เทียบกันไม่ได้
"""
import httpx
from app.config import settings


async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            json={"model": settings.embedding_model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
