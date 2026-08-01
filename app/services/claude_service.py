"""
ประกอบ prompt จาก chunks ที่ดึงมา + คำถามผู้ใช้ แล้วเรียก LLM ตอบ
รองรับสลับ provider ระหว่าง Claude กับ Groq ผ่าน settings.llm_provider
โดยไม่ต้องแก้โค้ดส่วนอื่น (webhook, retrieval เหมือนเดิมทุกอย่าง)
"""
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI  # groq ใช้ openai-compatible client
from app.config import settings
from app.models.schemas import RetrievedChunk

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับการซ่อมเครื่องจักร "
    "ใช้ข้อมูลจากคู่มือ/error code ที่ให้มาเท่านั้นในการตอบ "
    "ถ้าข้อมูลที่ให้มาไม่พอจะตอบ ให้บอกตรงๆ ว่าไม่มีข้อมูล ห้ามเดา"
)

_anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
_groq_client = AsyncOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")


def _build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(c.content for c in chunks)


async def ask_llm(question: str, chunks: list[RetrievedChunk]) -> str:
    context = _build_context(chunks)
    user_content = f"ข้อมูลอ้างอิงจากคู่มือ:\n{context}\n\nคำถามลูกค้า: {question}"

    if settings.llm_provider == "claude":
        resp = await _anthropic_client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return resp.content[0].text

    # default: groq
    resp = await _groq_client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content
