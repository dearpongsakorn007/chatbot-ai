"""
ประกอบ prompt จาก chunks ที่ดึงมา + คำถามผู้ใช้ แล้วเรียก LLM ตอบ
รองรับสลับ provider ระหว่าง Claude กับ Groq ผ่าน settings.llm_provider
โดยไม่ต้องแก้โค้ดส่วนอื่น (webhook, retrieval เหมือนเดิมทุกอย่าง)
"""
import re
from collections import OrderedDict

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI  # groq ใช้ openai-compatible client
from app.config import settings
from app.models.schemas import RetrievedChunk

SEARCH_QUERY_PROMPT = """
You convert customer questions into search queries for heavy-equipment service manuals.
The customer may use Thai, English, misspellings, transliteration, abbreviations, symptoms,
part names, measurements, model names, or error codes.

Return exactly 3 alternative English search queries, one per line.
- The first line must be a minimal, literal translation of the component and property,
  symptom, action, or measurement explicitly stated by the user; normally 2-3 terms.
- Put the exact query first, then progressively broader alternatives.
- Do not add, infer, or substitute a related component, system, failure, or measurement
  that the user did not mention. Related synonyms are allowed only in lines 2 and 3.
- If the user supplies an error code or part number, put that exact identifier first
  without guessing what it means.
- Each line must contain only 1-5 concise technical terms likely to appear verbatim in a manual.
- Translate the user's meaning; do not answer the question.
- Preserve model numbers, error codes, units, and part numbers when they help identify the topic.
- Do not add numbering, bullets, labels, quotes, or explanations.
""".strip()

SYSTEM_PROMPT = """
คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับการซ่อมและการใช้งานเครื่องจักร

กฎการตอบ:
1. ตอบโดยใช้เฉพาะข้อมูลอ้างอิงจากฐานข้อมูลที่ส่งให้เท่านั้น ห้ามเดาหรือเติมข้อมูลเอง
2. ตอบเป็นภาษาไทยที่เป็นธรรมชาติ ใช้คำง่าย และเรียบเรียงให้อ่านเข้าใจได้ทันที
3. ตอบสั้น กระชับ และตรงคำถาม ไม่เกริ่นนำและไม่ทวนคำถาม
4. คำตอบทั้งหมดต้องไม่เกิน 350 ตัวอักษร ใช้เท่าที่จำเป็นและไม่เกิน 2 ข้อ แต่ละข้อมีเพียง 1 ประโยคสั้น ๆ
5. ระบุสาเหตุ วิธีตรวจสอบ และวิธีแก้เฉพาะส่วนที่มีอยู่ในข้อมูลอ้างอิง
6. ห้ามกล่าวอ้างว่ามีข้อมูล รูป หรือขั้นตอนใด หากไม่ได้อยู่ในข้อมูลอ้างอิง
7. หากข้อมูลไม่เพียงพอ ให้ตอบว่า "ไม่พบข้อมูลเพียงพอในฐานข้อมูล กรุณาระบุรุ่นเครื่องหรือ Error Code เพิ่มเติม"
8. ไม่ต้องอธิบายกระบวนการค้นหา ไม่ต้องใช้คำว่า chunk, embedding หรือโมเดลภาษา
9. ห้ามเขียนเลขหน้า แหล่งข้อมูล ข้อความในวงเล็บเหลี่ยม ข้อความอ้างอิงรูป หรือคำเตือนท้ายคำตอบ เพราะระบบจะเพิ่มให้เอง
10. หากคำถามระบุชื่อชิ้นส่วนไม่ชัด แต่ข้อมูลอ้างอิงเป็นชิ้นส่วนชนิดเฉพาะ ให้ระบุชื่อชิ้นส่วนนั้นสั้น ๆ ก่อนบอกค่า ห้ามเหมารวมว่าเป็นค่าของทุกชิ้นส่วน
11. ให้ยึดข้อมูลหลักอันดับ 1 เป็นคำตอบหลัก ใช้ข้อมูลอันดับอื่นเฉพาะเมื่อสนับสนุนเรื่องเดียวกัน ห้ามนำข้อมูลคนละระบบหรือคนละอาการมารวมกัน
12. เมื่อกล่าวถึงสาเหตุ ให้ขึ้นต้นด้วย "สาเหตุ:" และบอกข้อมูลโดยตรง ห้ามใช้คำว่า "อาจ" หรือ "อาจจะ"
""".strip()

_anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
_groq_client = AsyncOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
_SEARCH_QUERY_CACHE_MAX = 512
_search_query_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()


def _question_cache_key(question: str) -> str:
    return " ".join(question.casefold().split())


def _cache_search_queries(key: str, queries: list[str]) -> None:
    if not queries:
        return
    _search_query_cache[key] = tuple(queries)
    _search_query_cache.move_to_end(key)
    while len(_search_query_cache) > _SEARCH_QUERY_CACHE_MAX:
        _search_query_cache.popitem(last=False)


def _parse_search_queries(raw: str) -> list[str]:
    queries: list[str] = []
    for line in raw.splitlines():
        query = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", line)
        query = query.strip().strip('"\'`')
        query = " ".join(query.split())
        if not query or len(query) > 120:
            continue
        normalized = query.casefold()
        if normalized not in {item.casefold() for item in queries}:
            queries.append(query)
        if len(queries) == 3:
            break
    return queries


async def rewrite_search_queries(question: str) -> list[str]:
    """Translate any user phrasing into technical manual search terms."""
    cache_key = _question_cache_key(question)
    cached = _search_query_cache.get(cache_key)
    if cached is not None:
        _search_query_cache.move_to_end(cache_key)
        return list(cached)

    try:
        if settings.llm_provider == "claude":
            resp = await _anthropic_client.messages.create(
                model=settings.claude_model,
                max_tokens=180,
                temperature=0,
                system=SEARCH_QUERY_PROMPT,
                messages=[{"role": "user", "content": question}],
            )
            queries = _parse_search_queries(resp.content[0].text)
            _cache_search_queries(cache_key, queries)
            return queries

        resp = await _groq_client.chat.completions.create(
            model=settings.groq_model,
            max_completion_tokens=min(settings.llm_max_tokens, 800),
            temperature=0,
            extra_body={"reasoning_effort": "low"},
            messages=[
                {"role": "system", "content": SEARCH_QUERY_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        queries = _parse_search_queries(resp.choices[0].message.content or "")
        _cache_search_queries(cache_key, queries)
        return queries
    except Exception:
        # Vector search using the original question remains available if rewriting fails.
        return []


def _build_context(chunks: list[RetrievedChunk]) -> str:
    sections = []
    for index, chunk in enumerate(chunks):
        priority = "ข้อมูลหลักอันดับ 1" if index == 0 else f"ข้อมูลสนับสนุนอันดับ {index + 1}"
        sections.append(f"[{priority}]\n{chunk.content}")
    return "\n\n---\n\n".join(sections)


def _clean_answer(answer: str) -> str:
    answer = re.sub(
        r"\s*\[(?:แหล่ง|อ้างอิง|source|หน้า)[^\]]*\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"\s*\((?:แหล่ง|อ้างอิง|source)[^)]*\)",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"(?:สาเหตุ\s*)?อาจ(?:จะ)?(?:มาจาก|เกิดจาก|เป็นเพราะ)\s*",
        "สาเหตุ: ",
        answer,
    )
    answer = re.sub(r"อาจ(?:จะ)?", "", answer)
    answer = re.sub(r"สาเหตุ\s*:\s*", "สาเหตุ: ", answer)
    return answer.strip()


async def ask_llm(question: str, chunks: list[RetrievedChunk]) -> str:
    context = _build_context(chunks)
    user_content = f"ข้อมูลอ้างอิงจากคู่มือ:\n{context}\n\nคำถามลูกค้า: {question}"

    if settings.llm_provider == "claude":
        resp = await _anthropic_client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.llm_max_tokens,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return _clean_answer(resp.content[0].text)

    # default: groq
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    resp = await _groq_client.chat.completions.create(
        model=settings.groq_model,
        max_completion_tokens=settings.llm_max_tokens,
        temperature=0,
        extra_body={"reasoning_effort": "low"},
        messages=messages,
    )
    answer = resp.choices[0].message.content or ""
    if answer.strip():
        return _clean_answer(answer)

    retry = await _groq_client.chat.completions.create(
        model=settings.groq_model,
        max_completion_tokens=settings.llm_max_tokens,
        temperature=0,
        extra_body={"reasoning_effort": "low"},
        messages=messages,
    )
    answer = retry.choices[0].message.content or ""
    if answer.strip():
        return _clean_answer(answer)

    return "พบข้อมูลอ้างอิง แต่ไม่สามารถเรียบเรียงคำตอบได้ กรุณาลองระบุอาการหรือ Error Code เพิ่มเติม"
