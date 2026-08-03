"""
ประกอบ prompt จาก chunks ที่ดึงมา + คำถามผู้ใช้ แล้วเรียก LLM ตอบ
รองรับสลับ provider ระหว่าง Claude กับ Groq ผ่าน settings.llm_provider
โดยไม่ต้องแก้โค้ดส่วนอื่น (webhook, retrieval เหมือนเดิมทุกอย่าง)
"""
import json
import re
from collections import OrderedDict
from dataclasses import dataclass

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
- Distinguish a component that physically detached or came loose from fluid leaking
  around that component. For physical detachment, search installation, clamp, holder,
  retaining bolt, tightening torque, loose, or mounting terms; do not translate it as
  a leak. Use the broader alternatives to cover both the retaining hardware and its
  tightening specification.
- Infer the fluid type only from the explicitly named system. In an injector or engine
  fuel-system question, Thai "น้ำมัน" means fuel/diesel, never hydraulic oil.
- Each line must contain only 1-5 concise technical terms likely to appear verbatim in a manual.
- Translate the user's meaning; do not answer the question.
- Preserve model numbers, error codes, units, and part numbers when they help identify the topic.
- Do not add numbering, bullets, labels, quotes, or explanations.
""".strip()

RERANK_PROMPT = """
You are a strict evidence gate for a heavy-equipment service manual.
Decide whether the candidates directly support the customer's exact physical symptom,
component, measurement, model, or error code.

Rules:
- If the wording can describe different failures that require different manual sections,
  return status "clarify" and ask one short Thai clarification question.
  Example ambiguity: a component physically came loose versus fluid leaked around it.
- For status "answer", select at most 2 candidates. Candidate 1 must contain the
  strongest direct evidence, and candidate 2 must support the same failure only.
- Reject generic pages, nearby topics, and pages about a different system or component.
- Matching only a machine model, component name, or broad symptom is not enough.
- A generic statement such as faulty injector or replace injector does not prove that
  an injector physically detached, came loose, or leaked. The evidence must state the
  same failure mode requested by the customer.
- Prefer explicit diagnostic steps, standard values, causes, and corrective actions.
- Every selection must include one short, contiguous, verbatim quote copied from that
  candidate which directly proves its relevance. Never paraphrase the evidence.
- Confidence must reflect direct support, not topical similarity.
- If no candidate directly supports the question, return status "none".

Return JSON only in exactly one of these shapes:
{"status":"clarify","clarification":"คำถามภาษาไทยสั้น ๆ","selections":[]}
{"status":"none","clarification":"","selections":[]}
{"status":"answer","clarification":"","selections":[{"index":2,"confidence":0.92,"evidence":"verbatim quote"}]}
""".strip()

SYSTEM_PROMPT = """
คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับการซ่อมและการใช้งานเครื่องจักร

กฎการตอบ:
1. ตอบโดยใช้เฉพาะข้อมูลอ้างอิงจากฐานข้อมูลที่ส่งให้เท่านั้น ห้ามเดาหรือเติมข้อมูลเอง
2. ตอบเป็นภาษาไทยที่เป็นธรรมชาติ ใช้คำง่าย และเรียบเรียงให้อ่านเข้าใจได้ทันที
3. ตอบสั้น กระชับ และตรงคำถาม ไม่เกริ่นนำและไม่ทวนคำถาม
4. คำตอบทั้งหมดต้องไม่เกิน 350 ตัวอักษร เขียนแต่ละหัวข้อคนละบรรทัดและไม่ใช้หัวข้อย่อยซ้อนกัน แต่ละหัวข้อต้องไม่เกิน 1 ประโยคสั้น
5. ใช้เฉพาะหัวข้อที่มีข้อมูลจริง ได้แก่ "สาเหตุ:", "ตรวจสอบ:", "ค่ามาตรฐาน:" และ "วิธีแก้:" หากไม่มีข้อมูลหัวข้อใดให้ตัดหัวข้อนั้นออก
6. หากข้อมูลมีหลายสาเหตุหรือหลายขั้นตอน ให้เลือกเฉพาะรายการที่ตรงคำถามและควรทำก่อนที่สุด ห้ามรวบรวมทุกความเป็นไปได้
7. ห้ามกล่าวอ้างว่ามีข้อมูล รูป หรือขั้นตอนใด หากไม่ได้อยู่ในข้อมูลอ้างอิง
8. หากข้อมูลไม่เพียงพอ ให้ตอบว่า "ไม่พบข้อมูลเพียงพอในฐานข้อมูล กรุณาระบุรุ่นเครื่องหรือ Error Code เพิ่มเติม"
9. ไม่ต้องอธิบายกระบวนการค้นหา ไม่ต้องใช้คำว่า chunk, embedding หรือโมเดลภาษา
10. ห้ามเขียนเลขหน้า แหล่งข้อมูล ข้อความในวงเล็บเหลี่ยม ข้อความอ้างอิงรูป หรือคำเตือนท้ายคำตอบ เพราะระบบจะเพิ่มให้เอง
11. หากคำถามระบุชื่อชิ้นส่วนไม่ชัด แต่ข้อมูลอ้างอิงเป็นชิ้นส่วนชนิดเฉพาะ ให้ระบุชื่อชิ้นส่วนนั้นสั้น ๆ ก่อนบอกค่า ห้ามเหมารวมว่าเป็นค่าของทุกชิ้นส่วน
12. ให้ยึดข้อมูลหลักอันดับ 1 เป็นคำตอบหลัก ใช้ข้อมูลอันดับอื่นเฉพาะเมื่อสนับสนุนเรื่องเดียวกัน ห้ามนำข้อมูลคนละระบบหรือคนละอาการมารวมกัน
13. เมื่อกล่าวถึงสาเหตุ ให้ขึ้นต้นด้วย "สาเหตุ:" และบอกข้อมูลโดยตรง ห้ามใช้คำว่า "อาจ" หรือ "อาจจะ"
""".strip()

_anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
_groq_client = AsyncOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
_SEARCH_QUERY_CACHE_MAX = 512
_MIN_EVIDENCE_CONFIDENCE = 0.8
_search_query_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()


@dataclass
class RerankResult:
    chunks: list[RetrievedChunk]
    clarification: str | None = None


def _question_cache_key(question: str) -> str:
    return " ".join(question.casefold().split())


def get_ambiguity_clarification(question: str) -> str | None:
    """Ask before retrieval when wording mixes fluid and physical detachment."""
    normalized = question.casefold()
    fluid_terms = ("น้ำมัน", "เชื้อเพลิง", "ดีเซล", "fuel", "diesel", " oil ")
    separation_terms = (
        "หลุด",
        "เด้ง",
        "ดันออก",
        "โผล่ออก",
        "loose",
        "detached",
        "popped out",
        "came out",
    )
    explicit_leak_terms = ("รั่ว", "ซึม", "ไหลออก", "leak", "seep")
    explicit_physical_terms = (
        "จากฝาสูบ",
        "ออกจากฝาสูบ",
        "แคลมป์",
        "ขายึด",
        "น็อตยึด",
        "clamp",
        "holder",
        "mounting bolt",
    )
    is_ambiguous = any(term in normalized for term in fluid_terms) and any(
        term in normalized for term in separation_terms
    )
    is_explicit = any(term in normalized for term in explicit_leak_terms) or any(
        term in normalized for term in explicit_physical_terms
    )
    if is_ambiguous and not is_explicit:
        return "หมายถึงตัวหัวฉีดหลุดออกจากฝาสูบ หรือน้ำมันรั่วออกบริเวณหัวฉีดหรือท่อครับ?"
    return None


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


def _normalize_evidence(text: str) -> str:
    return " ".join(text.casefold().split())


def _parse_rerank_result(raw: str, chunks: list[RetrievedChunk]) -> RerankResult:
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        payload = json.loads(raw[start:end])
    except (ValueError, TypeError, json.JSONDecodeError):
        return RerankResult(chunks=[])
    if not isinstance(payload, dict):
        return RerankResult(chunks=[])

    status = str(payload.get("status") or "").casefold()
    if status == "clarify":
        clarification = " ".join(str(payload.get("clarification") or "").split())
        if 10 <= len(clarification) <= 220:
            return RerankResult(chunks=[], clarification=clarification)
        return RerankResult(chunks=[])
    if status != "answer":
        return RerankResult(chunks=[])

    selected: list[RetrievedChunk] = []
    seen_indices: set[int] = set()
    for selection in payload.get("selections") or []:
        try:
            index = int(selection.get("index")) - 1
            confidence = float(selection.get("confidence"))
        except (AttributeError, TypeError, ValueError):
            continue
        evidence = " ".join(str(selection.get("evidence") or "").split())
        if (
            not 0 <= index < len(chunks)
            or index in seen_indices
            or confidence < _MIN_EVIDENCE_CONFIDENCE
            or len(evidence) < 20
        ):
            continue

        normalized_content = _normalize_evidence(chunks[index].content)
        normalized_evidence = _normalize_evidence(evidence)
        if normalized_evidence not in normalized_content:
            continue

        seen_indices.add(index)
        selected.append(chunks[index].model_copy(update={"content": evidence}))
        if len(selected) == 2:
            break
    return RerankResult(chunks=selected)


def _build_rerank_content(
    question: str,
    chunks: list[RetrievedChunk],
    search_queries: list[str] | None = None,
) -> str:
    candidates = []
    for index, chunk in enumerate(chunks, start=1):
        content = " ".join(chunk.content.split())[:5000]
        candidates.append(
            f"CANDIDATE {index}\n"
            f"Page: {chunk.reference or 'unknown'}\n"
            f"Content: {content}"
        )
    search_intent = " | ".join(search_queries or [])
    return (
        f"Customer question: {question}\n"
        f"Technical search intent: {search_intent}\n\n"
        + "\n\n---\n\n".join(candidates)
    )


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


async def rerank_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    search_queries: list[str] | None = None,
) -> RerankResult:
    """Use the LLM only as a strict relevance judge before answer generation."""
    if not chunks:
        return RerankResult(chunks=[])

    user_content = _build_rerank_content(question, chunks, search_queries)
    try:
        if settings.llm_provider == "claude":
            resp = await _anthropic_client.messages.create(
                model=settings.claude_model,
                max_tokens=80,
                temperature=0,
                system=RERANK_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = resp.content[0].text
        else:
            resp = await _groq_client.chat.completions.create(
                model=settings.groq_model,
                max_completion_tokens=min(settings.llm_max_tokens, 500),
                temperature=0,
                extra_body={"reasoning_effort": "low"},
                messages=[
                    {"role": "system", "content": RERANK_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            raw = resp.choices[0].message.content or ""
        return _parse_rerank_result(raw, chunks)
    except Exception:
        # Fail closed: do not answer or send an image without verified evidence.
        return RerankResult(chunks=[])


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
    answer = re.sub(r"(?<=\S)\s+สาเหตุ:\s*", " ", answer)

    compact_lines = []
    for raw_line in answer.splitlines():
        line = " ".join(raw_line.strip().lstrip("-•* ").split())
        if not line:
            continue

        # Keep only the first actionable item when the model returns a list.
        line = re.split(r"\s+2[.)]\s*", line, maxsplit=1)[0]
        line = re.sub(r"(?<=:)\s*1[.)]\s*", " ", line)
        if line.startswith(("ตรวจสอบ:", "วิธีแก้:")):
            line = re.split(
                r"\s+และ(?:ตรวจ|เช็ก|เช็ค|วัด|ปรับ|เปลี่ยน|ซ่อม)",
                line,
                maxsplit=1,
            )[0]
        if len(line) > 140:
            cut_at = max(
                line.rfind(".", 0, 141),
                line.rfind(";", 0, 141),
                line.rfind(",", 0, 141),
            )
            if cut_at >= 70:
                line = line[: cut_at + 1]
            else:
                boundaries = [
                    line.find(separator, 60, 141)
                    for separator in (" และ", " หรือ", " ซึ่ง", " โดย", " ทำให้")
                ]
                boundaries = [position for position in boundaries if position >= 60]
                if boundaries:
                    line = line[: min(boundaries)].rstrip(" ,;:-")
                else:
                    cut_at = line.rfind(" ", 0, 138)
                    if cut_at < 70:
                        cut_at = 137
                    line = line[:cut_at].rstrip(" ,;:-") + "."
        compact_lines.append(line)
        if len(compact_lines) == 4:
            break

    selected_lines = []
    current_length = 0
    for line in compact_lines:
        added_length = len(line) + (1 if selected_lines else 0)
        if selected_lines and current_length + added_length > 350:
            break
        selected_lines.append(line)
        current_length += added_length
    compact = "\n".join(selected_lines)
    return compact.strip()


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
