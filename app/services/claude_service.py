"""
ประกอบ prompt จาก chunks ที่ดึงมา + คำถามผู้ใช้ แล้วเรียก LLM ตอบ
รองรับสลับ provider ระหว่าง Claude กับ Groq ผ่าน settings.llm_provider
โดยไม่ต้องแก้โค้ดส่วนอื่น (webhook, retrieval เหมือนเดิมทุกอย่าง)
"""
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI  # groq ใช้ openai-compatible client
from app.config import settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger("repair-bot")

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
- This corpus is already limited to the selected machine model. Never return the model name
  plus only generic words such as manual, service, diagnostic, repair, or maintenance.
- A label such as "DIS 3", "display 3", or a screen/channel number is not an error code.
  Search the component, measured property, operating symptom, and test condition shown there.
- Every line must still contain a useful component, property, symptom, action, value, or exact
  identifier after the machine model is removed.
- Do not add numbering, bullets, labels, quotes, or explanations.
""".strip()

RERANK_PROMPT = """
You are a strict evidence gate for a heavy-equipment service manual.
Decide whether the candidates directly support the customer's exact physical symptom,
component, measurement, model, or error code.

Rules:
- Never ask a clarification question. When wording can describe different failures, select
  the candidate that supports the most explicit component, measurement, identifier, or
  physical symptom and keep the answer within that supported scope.
- For status "answer", select at most 2 candidates. Candidate 1 must contain the
  strongest direct evidence, and candidate 2 must support the same failure only.
- Reject generic pages, nearby topics, and pages about a different system or component.
- Matching only a machine model, component name, or broad symptom is not enough.
- Do not map a whole-machine symptom such as generally slow or low power to one specific
  actuator/function (boom, arm, bucket, swing, or travel) unless the customer explicitly
  named that function. Prefer a general pump/engine diagnostic page for a general symptom.
- A generic statement such as faulty injector or replace injector does not prove that
  an injector physically detached, came loose, or leaked. The evidence must state the
  same failure mode requested by the customer.
- Prefer explicit diagnostic steps, standard values, causes, and corrective actions.
- Judge the complete intent, not one matching keyword. For measurements, require the
  same component/property and prefer candidates that state the test condition and unit.
  For symptoms, prefer candidates that contain an ordered diagnostic path. For procedures,
  parts, wiring, and error codes, require the same action, component/circuit, or identifier.
- Prefer a candidate containing the answer-bearing table row or paragraph over a nearby
  index, heading, generic maintenance page, or page that only names the component.
- A corrective action is direct evidence only when the candidate explicitly associates
  it with the matching confirmed condition. A list of checks is not proof that the first
  listed part has failed.
- For a compound customer question, a candidate may directly answer one explicit part of
  the question (for example the reported measurement and its diagnostic procedure) even
  when it does not prove the downstream symptom. Select it and let the answer stay within
  that supported scope; do not return "none" merely because one page cannot prove every clause.
- When the customer reports an abnormal value, a page for the same component/property that
  provides its test condition, expected range, or ordered checks is direct evidence. It does
  not need to repeat the customer's exact numeric value or operational symptom.
- Treat standalone monitor labels such as "DIS 3", "display 3", "screen 3", or a channel
  number as a readout position, not an error code or failure-type number, unless the customer
  explicitly says it is an error code. Never select an error-code table only because its
  number matches the display/channel number.
- Every selection must include short verbatim evidence copied from that candidate which
  directly proves its relevance. For prose, use one contiguous quote. For a structured
  table row, copy 2-4 exact cells in reading order and separate them with " | ". Never
  paraphrase cells and never use ellipses to replace meaningful text.
- Confidence must reflect direct support, not topical similarity.
- If no candidate directly supports the question, return status "none".

Return JSON only in exactly one of these shapes:
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
6. เลือกข้อมูลตามเจตนาคำถาม: อาการเสียให้เรียงจุดตรวจตามคู่มือ, Error Code ให้บอกความหมาย/เงื่อนไข/จุดตรวจ, ค่ามาตรฐานให้ระบุชิ้นส่วน/เงื่อนไข/ค่าและหน่วย, วิธีซ่อมให้รักษาลำดับขั้นตอน, อะไหล่และวงจรไฟฟ้าให้ระบุชิ้นส่วนหรือขั้วที่ตรงคำถาม
7. ห้ามกล่าวอ้างว่ามีข้อมูล รูป หรือขั้นตอนใด หากไม่ได้อยู่ในข้อมูลอ้างอิง
8. หากข้อมูลไม่เพียงพอ ให้ตอบว่า "ไม่พบข้อมูลเพียงพอในฐานข้อมูล กรุณาระบุรุ่นเครื่องหรือ Error Code เพิ่มเติม"
9. ไม่ต้องอธิบายกระบวนการค้นหา ไม่ต้องใช้คำว่า chunk, embedding หรือโมเดลภาษา
10. ห้ามเขียนเลขหน้า แหล่งข้อมูล ข้อความในวงเล็บเหลี่ยม ข้อความอ้างอิงรูป หรือคำเตือนท้ายคำตอบ เพราะระบบจะเพิ่มให้เอง
11. หากคำถามระบุชื่อชิ้นส่วนไม่ชัด แต่ข้อมูลอ้างอิงเป็นชิ้นส่วนชนิดเฉพาะ ให้ระบุชื่อชิ้นส่วนนั้นสั้น ๆ ก่อนบอกค่า ห้ามเหมารวมว่าเป็นค่าของทุกชิ้นส่วน
12. ให้ยึดข้อมูลหลักอันดับ 1 เป็นคำตอบหลัก ใช้ข้อมูลอันดับอื่นเฉพาะเมื่อสนับสนุนเรื่องเดียวกัน ห้ามนำข้อมูลคนละระบบหรือคนละอาการมารวมกัน
13. ใช้หัวข้อ "สาเหตุ:" เฉพาะเมื่อข้อมูลอ้างอิงระบุความสัมพันธ์ว่าเป็นสาเหตุโดยตรง ห้ามเปลี่ยนรายการตรวจสอบให้กลายเป็นข้อสรุปว่าชิ้นส่วนเสีย
14. หากคู่มือให้ตรวจหลายจุดก่อนวินิจฉัย ให้ใช้หัวข้อ "ตรวจสอบ:" และเรียงรายการสำคัญตามลำดับในคู่มือไม่เกิน 4 รายการ ห้ามสรุปให้เปลี่ยนอะไหล่ตัวแรกทันที
15. ใช้หัวข้อ "วิธีแก้:" เฉพาะเมื่อข้อมูลอ้างอิงระบุการแก้ไขสำหรับเงื่อนไขที่ยืนยันแล้วโดยตรง หากยังเป็นเพียงขั้นตอนวินิจฉัยให้แสดงเฉพาะ "ตรวจสอบ:"
15.1 ข้อความว่า "ตรวจว่าชิ้นส่วนทำงานปกติ" เป็นเพียงขั้นตอนตรวจสอบ ไม่ได้อนุญาตให้สรุปว่าเสียหรือให้เปลี่ยนชิ้นส่วนนั้น
16. ค่ามาตรฐานต้องคงตัวเลข หน่วย ชิ้นส่วน และเงื่อนไขการวัดจากคู่มือ ห้ามนำค่าจากคนละโหมดหรือคนละเงื่อนไขมาเปรียบเทียบกัน
17. หลักฐานที่ยืนยันเป็นเพียงจุดบอกว่าหน้านี้ตรงคำถาม ให้ใช้เนื้อหาเต็มของข้อมูลหลักเพื่อรักษาบริบทและลำดับ แต่ห้ามนำหัวข้อข้างเคียงที่ไม่เกี่ยวข้องมาตอบ
18. เมื่อกล่าวถึงสาเหตุ ให้ขึ้นต้นด้วย "สาเหตุ:" และบอกข้อมูลโดยตรง ห้ามใช้คำว่า "อาจ" หรือ "อาจจะ"
19. ทุกบรรทัดของคำตอบต้องลงท้ายด้วยคำว่า "ครับ"
20. ใช้คำศัพท์ช่างให้คงที่: relief valve = วาล์วระบายแรงดัน, pump proportional valve = วาล์วสัดส่วนปั๊ม และ pump regulator = เรกูเลเตอร์ปั๊ม ห้ามแปล relief valve เป็นลิฟท์วาล์ว
""".strip()

_anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
_groq_client = AsyncOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
_SEARCH_QUERY_CACHE_MAX = 512
_MIN_EVIDENCE_CONFIDENCE = 0.65
_SAFE_FALLBACK_MIN_SUPPORT = 2
_search_query_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()
_FUNCTION_SCOPE_TERMS = {
    "boom": ("boom", "บูม"),
    "arm": ("arm", "อาร์ม"),
    "bucket": ("bucket", "บุ้งกี๋", "บุ้งกี๊"),
    "swing": ("swing", "สวิง"),
    "travel": ("travel", "เดิน", "ตีนตะขาบ"),
}


@dataclass
class RerankResult:
    chunks: list[RetrievedChunk]
    clarification: str | None = None
    reason: str = "unknown"


def _question_cache_key(question: str) -> str:
    return " ".join(question.casefold().split())


def get_ambiguity_clarification(question: str) -> str | None:
    """Compatibility hook: technical questions now continue without asking back."""
    return None


# รูปแบบรหัส error ของ KOBELCO SK200-8 เช่น A015, B012, D063
_ERROR_CODE_PATTERN = re.compile(r"\b[A-Za-z]\d{3,4}\b")
_ERROR_CODE_KEYWORDS = (
    "error code",
    "errorcode",
    "โค้ด",
    "รหัสเออเรอร์",
    "รหัส error",
    "code error",
    "error",
)
ERROR_CODE_CONTENT_TYPES = [
    "error_code",
    "error_code_detail",
    "error_code_table",
    "error_code_index",
]

_MEASUREMENT_KEYWORDS = (
    "ค่ามาตรฐาน", "วัดค่า", "แรงดัน", "แรงบิด", "โวลต์", "โอห์ม", "แอมป์", "อุณหภูมิ",
    "standard value", "measurement", "pressure", "torque", "voltage", "resistance", "specification",
)
_SYMPTOM_KEYWORDS = (
    "อาการ", "เสีย", "ไม่ทำงาน", "ไม่มีแรง", "ช้า", "รั่ว", "เสียง", "ร้อน", "ดับ", "สตาร์ท",
    "เกิดจาก", "สาเหตุ", "trouble", "fault", "slow", "leak", "noise", "overheat", "no power",
)
_PROCEDURE_KEYWORDS = (
    "วิธี", "ขั้นตอน", "ตรวจสอบ", "ถอด", "ประกอบ", "เปลี่ยน", "ติดตั้ง", "ปรับตั้ง",
    "how to", "procedure", "inspection", "replace", "install", "remove", "adjust",
)
_PART_KEYWORDS = ("อะไหล่", "ชิ้นส่วน", "หมายเลขอะไหล่", "part number", "component")


def infer_content_type_filter(question: str) -> list[str] | None:
    """จำกัดการค้นหาเฉพาะหมวด error code เมื่อคำถามระบุรหัส/พูดถึง error code ชัดเจน
    ป้องกันไม่ให้ผลลัพธ์ปนกับ chunk หมวดอื่น (procedure/table/narrative) ที่ไม่เกี่ยวกับรหัสที่ถาม
    คืนค่า None เมื่อไม่ชัดเจนพอ เพื่อไม่ให้จำกัดผลค้นหาผิดสำหรับคำถามทั่วไป
    """
    normalized = question.casefold()
    has_code_pattern = bool(_ERROR_CODE_PATTERN.search(question))
    has_keyword = any(keyword in normalized for keyword in _ERROR_CODE_KEYWORDS)
    if has_code_pattern or has_keyword:
        return list(ERROR_CODE_CONTENT_TYPES)
    return None


def infer_search_category_hint(question: str) -> str | None:
    """ส่งคำใบ้หมวดหมู่ให้ vector search โดยไม่ใช้เป็น hard filter."""
    normalized = question.casefold()
    if infer_content_type_filter(question):
        return "error code diagnosis cause remedy"
    if any(keyword in normalized for keyword in _MEASUREMENT_KEYWORDS):
        return "standard value measurement inspection"
    if any(keyword in normalized for keyword in _SYMPTOM_KEYWORDS):
        return "troubleshooting cause diagnosis remedy"
    if any(keyword in normalized for keyword in _PROCEDURE_KEYWORDS):
        return "procedure inspection adjustment"
    if any(keyword in normalized for keyword in _PART_KEYWORDS):
        return "component part number removal installation"
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
        query = _normalize_search_query(query)
        if not query or len(query) > 120 or _is_low_information_search_query(query):
            continue
        normalized = query.casefold()
        if normalized not in {item.casefold() for item in queries}:
            queries.append(query)
        if len(queries) == 3:
            break
    return queries


_GENERIC_SEARCH_TERMS = {
    "diagnostic", "diagnosis", "display", "dis", "manual", "service", "repair",
    "maintenance", "procedure", "system", "machine", "excavator", "kobelco",
}


def _normalize_search_query(query: str) -> str:
    """Remove UI labels and normalize common indexed-component notation."""
    query = re.sub(
        r"\b(?:dis|display|screen|channel)\s*[-:#]?\s*\d+\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(
        r"\bpump\s*[-#]?\s*([12])\b",
        lambda match: f"P{match.group(1)} pump pressure",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(r"\bpressure\s+pressure\b", "pressure", query, flags=re.IGNORECASE)
    return " ".join(query.split())


def _is_low_information_search_query(query: str) -> bool:
    """Reject model-only/manual queries which overwhelm full-text rank fusion."""
    normalized = query.casefold()
    if _ERROR_CODE_PATTERN.search(query):
        return False
    normalized = re.sub(r"\b(?:kobelco\s*)?sk\s*[- ]?\d{2,4}(?:-\d+)?\b", " ", normalized)
    tokens = re.findall(r"[a-z0-9][a-z0-9_-]*", normalized)
    useful_tokens = []
    has_specific_identifier = False
    for token in tokens:
        plain = token.strip("_-")
        if not plain or plain.isdigit() or plain in _GENERIC_SEARCH_TERMS:
            continue
        has_letter = any(char.isalpha() for char in plain)
        has_digit = any(char.isdigit() for char in plain)
        if has_letter and has_digit and len(plain) >= 5:
            has_specific_identifier = True
        if (has_letter and len(plain) >= 3) or (has_letter and has_digit and len(plain) >= 4):
            useful_tokens.append(plain)
    return not has_specific_identifier and len(useful_tokens) < 2


def _normalize_evidence(text: str) -> str:
    return " ".join(text.casefold().split())


def _search_terms(search_queries: list[str] | None) -> set[str]:
    terms: set[str] = set()
    for query in search_queries or []:
        normalized = re.sub(
            r"\b(?:kobelco\s*)?sk\s*[- ]?\d{2,4}(?:-\d+)?\b",
            " ",
            query.casefold(),
        )
        for token in re.findall(r"[a-z0-9][a-z0-9_-]*", normalized):
            token = token.strip("_-")
            if (
                not token
                or token.isdigit()
                or token in _GENERIC_SEARCH_TERMS
                or (len(token) < 3 and token not in {"p1", "p2"})
            ):
                continue
            terms.add(token)
    return terms


def _candidate_support_score(
    chunk: RetrievedChunk,
    search_queries: list[str] | None,
) -> int:
    """Deterministic lexical support used when the LLM quote is malformed."""
    terms = _search_terms(search_queries)
    if not terms:
        return 0
    content = chunk.content.casefold()
    matched = {term for term in terms if term in content}
    score = len(matched)
    if any(
        any(char.isalpha() for char in term)
        and any(char.isdigit() for char in term)
        and len(term) >= 4
        and term in matched
        for term in terms
    ):
        score += 2
    return score


def _select_safe_fallback(
    chunks: list[RetrievedChunk],
    search_queries: list[str] | None,
    allow_top_ranked: bool = False,
) -> RetrievedChunk | None:
    scored = [
        (_candidate_support_score(chunk, search_queries), index, chunk)
        for index, chunk in enumerate(chunks)
    ]
    if not scored:
        return None
    support, _, chunk = max(scored, key=lambda item: (item[0], -item[1]))
    if support < _SAFE_FALLBACK_MIN_SUPPORT:
        if allow_top_ranked:
            top_chunk = chunks[0]
            if top_chunk.content.strip():
                return top_chunk.model_copy(update={"verified_evidence": None})
        return None
    return chunk.model_copy(update={"verified_evidence": None})


def _evidence_is_supported(content: str, evidence: str) -> bool:
    """Verify either one prose quote or ordered verbatim cells from an OCR table."""
    normalized_content = _normalize_evidence(content)
    normalized_evidence = _normalize_evidence(evidence)
    if normalized_evidence in normalized_content:
        return True

    raw_segments = re.split(r"\s*(?:\|+|\.{3,}|…+)\s*", evidence)
    segments = [
        _normalize_evidence(segment.strip(" |.;"))
        for segment in raw_segments
        if len(_normalize_evidence(segment.strip(" |.;"))) >= 8
    ]
    if len(segments) < 2:
        return False

    position = 0
    ordered = True
    for segment in segments:
        found_at = normalized_content.find(segment, position)
        if found_at < 0:
            ordered = False
            break
        position = found_at + len(segment)
    if ordered:
        return True

    # Models sometimes reorder verified table cells to express symptom -> condition.
    # Still require every complete cell to exist verbatim on the same selected page.
    return all(segment in normalized_content for segment in segments)


def _parse_rerank_result(
    raw: str,
    chunks: list[RetrievedChunk],
    search_queries: list[str] | None = None,
) -> RerankResult:
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        payload = json.loads(raw[start:end])
    except (ValueError, TypeError, json.JSONDecodeError):
        return RerankResult(chunks=[], reason="invalid_json")
    if not isinstance(payload, dict):
        return RerankResult(chunks=[], reason="invalid_payload")

    status = str(payload.get("status") or "").casefold()
    if status == "clarify":
        return RerankResult(chunks=[], reason="model_requested_clarification")
    if status != "answer":
        return RerankResult(chunks=[], reason=f"status_{status or 'missing'}")

    selected: list[RetrievedChunk] = []
    seen_indices: set[int] = set()
    accepted_by_semantic_support = False
    rejection_reasons: set[str] = set()
    for selection in payload.get("selections") or []:
        try:
            index = int(selection.get("index")) - 1
            confidence = float(selection.get("confidence"))
        except (AttributeError, TypeError, ValueError):
            rejection_reasons.add("invalid_selection")
            continue
        evidence = " ".join(str(selection.get("evidence") or "").split())
        if not 0 <= index < len(chunks) or index in seen_indices:
            rejection_reasons.add("invalid_index")
            continue
        if confidence < _MIN_EVIDENCE_CONFIDENCE:
            rejection_reasons.add("low_confidence")
            continue

        evidence_supported = len(evidence) >= 20 and _evidence_is_supported(
            chunks[index].content, evidence
        )
        semantic_support = _candidate_support_score(chunks[index], search_queries)
        if not evidence_supported and semantic_support < _SAFE_FALLBACK_MIN_SUPPORT:
            rejection_reasons.add("quote_mismatch")
            continue
        accepted_by_semantic_support = accepted_by_semantic_support or not evidence_supported

        seen_indices.add(index)
        # Preserve the complete selected passage for answer generation. The short quote
        # only proves relevance; replacing the passage with it loses conditions, values,
        # and the ordered checks that surround the quote.
        selected.append(
            chunks[index].model_copy(
                update={"verified_evidence": evidence if evidence_supported else None}
            )
        )
        if len(selected) == 2:
            break
    if selected:
        reason = (
            "selected_semantic_support"
            if accepted_by_semantic_support
            else "selected_verified_quote"
        )
        return RerankResult(chunks=selected, reason=reason)
    reason = "+".join(sorted(rejection_reasons)) or "no_selections"
    return RerankResult(chunks=[], reason=reason)


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


def _text_has_scope_term(text: str, term: str) -> bool:
    if term.isascii():
        return bool(re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE))
    return term in text


def _filter_scope_mismatches(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Do not map a general symptom to a specific actuator the user never named."""
    question_scopes = {
        scope
        for scope, terms in _FUNCTION_SCOPE_TERMS.items()
        if any(_text_has_scope_term(question, term) for term in terms)
    }
    filtered = []
    for chunk in chunks:
        # Metadata, heading, and first row identify the page's functional scope.
        heading_area = chunk.content[:2000]
        chunk_scopes = {
            scope
            for scope, terms in _FUNCTION_SCOPE_TERMS.items()
            if any(_text_has_scope_term(heading_area, term) for term in terms)
        }
        if chunk_scopes and not chunk_scopes.intersection(question_scopes):
            continue
        filtered.append(chunk)
    return filtered


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
    original_chunks = chunks
    input_count = len(original_chunks)
    chunks = _filter_scope_mismatches(question, original_chunks)
    if not chunks:
        fallback = _select_safe_fallback(
            original_chunks,
            search_queries,
            allow_top_ranked=True,
        )
        logger.info(
            "rerank completed input=%d scoped=0 selected=%d reason=scope_mismatch fallback=%s",
            input_count,
            int(bool(fallback)),
            str(bool(fallback)).lower(),
        )
        if fallback:
            return RerankResult(
                chunks=[fallback], reason="fallback_after_scope_mismatch"
            )
        return RerankResult(chunks=[], reason="scope_mismatch_no_candidates")

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
        result = _parse_rerank_result(raw, chunks, search_queries)
        used_fallback = False
        if not result.chunks:
            fallback = _select_safe_fallback(
                chunks,
                search_queries,
                allow_top_ranked=True,
            )
            if fallback:
                result = RerankResult(
                    chunks=[fallback], reason=f"fallback_after_{result.reason}"
                )
                used_fallback = True
        logger.info(
            "rerank completed input=%d scoped=%d selected=%d reason=%s fallback=%s",
            input_count,
            len(chunks),
            len(result.chunks),
            result.reason,
            str(used_fallback).lower(),
        )
        return result
    except Exception as exc:
        fallback = _select_safe_fallback(
            chunks,
            search_queries,
            allow_top_ranked=True,
        )
        logger.warning(
            "rerank failed error_type=%s input=%d scoped=%d fallback=%s",
            type(exc).__name__,
            input_count,
            len(chunks),
            str(bool(fallback)).lower(),
        )
        if fallback:
            return RerankResult(chunks=[fallback], reason="fallback_after_error")
        return RerankResult(chunks=[], reason="rerank_error")


def _build_context(chunks: list[RetrievedChunk]) -> str:
    sections = []
    for index, chunk in enumerate(chunks):
        priority = "ข้อมูลหลักอันดับ 1" if index == 0 else f"ข้อมูลสนับสนุนอันดับ {index + 1}"
        evidence = chunk.verified_evidence or "ผ่านการตรวจคำศัพท์และขอบเขตหน้า"
        content = chunk.content.strip()[:6000]
        sections.append(
            f"[{priority}]\n"
            f"หลักฐานยืนยันความเกี่ยวข้อง: {evidence}\n"
            f"เนื้อหาคู่มือฉบับเต็มของรายการนี้:\n{content}"
        )
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

        # Keep a concise ordered diagnostic path instead of silently discarding every
        # check after item 1. The prompt limits the list to four relevant items.
        line = re.sub(r"(?<=:)\s*1[.)]\s*", " 1) ", line)
        if len(line) > 220:
            cut_at = max(
                line.rfind(".", 0, 221),
                line.rfind(";", 0, 221),
                line.rfind(",", 0, 221),
            )
            if cut_at >= 120:
                line = line[: cut_at + 1]
            else:
                boundaries = [
                    line.find(separator, 120, 221)
                    for separator in (" และ", " หรือ", " ซึ่ง", " โดย", " ทำให้")
                ]
                boundaries = [position for position in boundaries if position >= 120]
                if boundaries:
                    line = line[: min(boundaries)].rstrip(" ,;:-")
                else:
                    cut_at = line.rfind(" ", 0, 218)
                    if cut_at < 120:
                        cut_at = 217
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
