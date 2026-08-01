"""
ประกอบ prompt จาก chunks ที่ดึงมา + คำถามผู้ใช้ แล้วเรียก LLM ตอบ
รองรับสลับ provider ระหว่าง Claude กับ Groq ผ่าน settings.llm_provider
โดยไม่ต้องแก้โค้ดส่วนอื่น (webhook, retrieval เหมือนเดิมทุกอย่าง)
"""
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI  # groq ใช้ openai-compatible client
from app.config import settings
from app.models.schemas import RetrievedChunk

SYSTEM_PROMPT = """
คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับการซ่อมและการใช้งานเครื่องจักร

กฎการตอบ:
1. ตอบโดยใช้เฉพาะข้อมูลอ้างอิงจากฐานข้อมูลที่ส่งให้เท่านั้น ห้ามเดาหรือเติมข้อมูลเอง
2. ตอบเป็นภาษาไทยที่เป็นธรรมชาติ ใช้คำง่าย และเรียบเรียงให้อ่านเข้าใจได้ทันที
3. ตอบสั้น กระชับ และตรงคำถาม ไม่เกริ่นนำและไม่ทวนคำถาม
4. คำตอบทั่วไปให้ตอบ 1-3 ประโยค หากมีขั้นตอนให้ใช้ไม่เกิน 3 ข้อสั้น ๆ
5. ระบุสาเหตุ วิธีตรวจสอบ และวิธีแก้เฉพาะส่วนที่มีอยู่ในข้อมูลอ้างอิง
6. ห้ามกล่าวอ้างว่ามีข้อมูล รูป หรือขั้นตอนใด หากไม่ได้อยู่ในข้อมูลอ้างอิง
7. หากข้อมูลไม่เพียงพอ ให้ตอบว่า "ไม่พบข้อมูลเพียงพอในฐานข้อมูล กรุณาระบุรุ่นเครื่องหรือ Error Code เพิ่มเติม"
8. ไม่ต้องอธิบายกระบวนการค้นหา ไม่ต้องใช้คำว่า chunk, embedding หรือโมเดลภาษา
9. ไม่ต้องเขียนข้อความอ้างอิงรูปหรือคำเตือนท้ายคำตอบ เพราะระบบจะเพิ่มให้เอง
""".strip()

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
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return resp.content[0].text

    # default: groq
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    resp = await _groq_client.chat.completions.create(
        model=settings.groq_model,
        max_completion_tokens=500,
        extra_body={"reasoning_effort": "low"},
        messages=messages,
    )
    answer = resp.choices[0].message.content or ""
    if answer.strip():
        return answer.strip()

    retry = await _groq_client.chat.completions.create(
        model=settings.groq_model,
        max_completion_tokens=800,
        extra_body={"reasoning_effort": "low"},
        messages=messages,
    )
    answer = retry.choices[0].message.content or ""
    if answer.strip():
        return answer.strip()

    return "พบข้อมูลอ้างอิง แต่ไม่สามารถเรียบเรียงคำตอบได้ กรุณาลองระบุอาการหรือ Error Code เพิ่มเติม"
