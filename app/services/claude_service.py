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
- For status "answer", select at most 3 candidates. Candidate 1 must contain the
  strongest direct evidence, and candidates 2-3 must support the same failure only.
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
- Some tables list named failure symptoms as rows ("Row group:") against shared checking
  factors as columns (a troubleshooting matrix). For this table type, a candidate is direct
  evidence only when the customer's exact described symptom matches one of the table's named
  row symptoms, or an unambiguous close paraphrase of it. A checking factor that repeats
  across many unrelated rows (e.g. "leak, clogging of fuel system") is never evidence for a
  symptom that is absent from the table, even if it echoes words from the question.
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

CLARIFYING_QUESTION_PROMPT = """
ระบบค้นข้อมูลไม่พบเนื้อหาที่ตรงกับคำถามลูกค้าเกี่ยวกับการซ่อมเครื่องจักร KOBELCO SK200-8
ตั้งคำถามกลับสั้นๆ ภาษาไทย 1 ประโยค เพื่อให้ลูกค้าระบุข้อมูลที่ขาดหายให้ชัดเจนขึ้น
เช่น ชื่อชิ้นส่วน/ระบบที่แน่ชัด ตำแหน่งที่เกิดอาการ หรือความหมายของคำศัพท์คลุมเครือในคำถามลูกค้า

กฎ:
- อ้างอิงคำศัพท์ที่ลูกค้าใช้จริงในคำถาม เพื่อถามให้ตรงจุดที่ขาดหาย ไม่ใช่คำถามทั่วไปลอยๆ
- ห้ามพูดถึงฐานข้อมูล คู่มือ หรือกระบวนการค้นหาใดๆ
- ห้ามขอโทษ ห้ามพูดว่า "ไม่พบข้อมูล" (ระบบจะเติมให้เอง)
- ตอบเป็นคำถามเดียวประโยคเดียว ไม่เกิน 150 ตัวอักษร ลงท้ายด้วย "ครับ"
- ถ้าคำถามลูกค้าสั้น/กว้างมากจนไม่มีจุดเฉพาะให้จับ ให้ถามขอชื่อชิ้นส่วนหรืออาการที่ชัดเจนขึ้นแทน
- ห้ามใส่คำนำ คำอธิบาย หรือเครื่องหมายคำพูดคร่อมประโยค
""".strip()

SYSTEM_PROMPT = """
คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับการซ่อมและการใช้งานเครื่องจักร

กฎการตอบ:
1. ตอบโดยใช้เฉพาะข้อมูลอ้างอิงจากฐานข้อมูลที่ส่งให้เท่านั้น ห้ามเดาหรือเติมข้อมูลเอง
2. ตอบเป็นภาษาไทยธรรมชาติ แบบช่างเทคนิคอธิบายให้ลูกค้าฟังปากเปล่า ไม่ใช่รายงานที่มีหัวข้อ
2.1 แปลข้อมูลอ้างอิงที่เป็นภาษาอังกฤษเป็นภาษาไทยทั้งหมดเสมอ ห้ามคงชื่อขั้นตอน หัวข้อย่อย หรือประโยคภาษาอังกฤษ
    ไว้ในเครื่องหมายคำพูดโดยไม่แปล (เช่น ห้ามเขียน "Bleed air in pump" ให้แปลว่า "ไล่ลมออกจากปั๊ม")
    ยกเว้นสิ่งที่ต้องคงไว้ตามต้นฉบับเพราะเป็นตัวระบุเฉพาะ ได้แก่ รุ่นเครื่อง รหัสอะไหล่ รหัส error code
    ชื่อขั้ว/คอนเนคเตอร์ และตัวเลข/หน่วยวัด
3. ห้ามขึ้นต้นประโยคหรือส่วนใดของคำตอบด้วยหัวข้อ/label เช่น "สาเหตุ:", "ตรวจสอบ:", "ค่ามาตรฐาน:", "วิธีแก้:"
   หรือคำอื่นที่ทำหน้าที่เป็นหัวข้อคล้ายกัน ให้ร้อยเรียงเนื้อหาทั้งหมดเป็นข้อความเดียวไหลต่อเนื่องกันเหมือนพูดคุยจริง
   ไม่เกริ่นนำ ไม่ทวนคำถาม
4. คำตอบทั้งหมดต้องไม่เกิน 600 ตัวอักษร เขียนเป็นข้อความเดียว ไม่ขึ้นบรรทัดใหม่โดยไม่จำเป็น
   ถ้าต้องแจกแจงขั้นตอนตรวจสอบหลายข้อ ให้ใส่เลข 1) 2) 3) ต่อกันในประโยคเดียวกันได้ ไม่ต้องแยกบรรทัด
5. พูดถึงเฉพาะประเด็นที่มีข้อมูลจริงรองรับ (เช่น สาเหตุ, สิ่งที่ต้องตรวจ, ค่ามาตรฐาน, วิธีแก้) โดยไม่ต้องประกาศชื่อหัวข้อ
   เรียงลำดับให้อ่านลื่นตามธรรมชาติ ถ้าไม่มีข้อมูลด้านไหนก็ข้ามด้านนั้นไปเลย ไม่ต้องพยายามพูดให้ครบทุกด้าน
6. เลือกข้อมูลตามเจตนาคำถาม: อาการเสียให้เรียงจุดตรวจตามคู่มือ, Error Code ให้บอกความหมาย/เงื่อนไข/จุดตรวจ, ค่ามาตรฐานให้ระบุชิ้นส่วน/เงื่อนไข/ค่าและหน่วย, วิธีซ่อมให้รักษาลำดับขั้นตอน, อะไหล่และวงจรไฟฟ้าให้ระบุชิ้นส่วนหรือขั้วที่ตรงคำถาม
7. ห้ามกล่าวอ้างว่ามีข้อมูล รูป หรือขั้นตอนใด หากไม่ได้อยู่ในข้อมูลอ้างอิง
8. หากข้อมูลไม่เพียงพอ ให้ตอบว่า "ไม่พบข้อมูลเพียงพอในฐานข้อมูล กรุณาระบุรุ่นเครื่องหรือ Error Code เพิ่มเติม"
9. ไม่ต้องอธิบายกระบวนการค้นหา ไม่ต้องใช้คำว่า chunk, embedding หรือโมเดลภาษา
10. ห้ามเขียนเลขหน้า แหล่งข้อมูล ข้อความในวงเล็บเหลี่ยม ข้อความอ้างอิงรูป หรือคำเตือนท้ายคำตอบ เพราะระบบจะเพิ่มให้เอง
11. หากคำถามระบุชื่อชิ้นส่วนไม่ชัด แต่ข้อมูลอ้างอิงเป็นชิ้นส่วนชนิดเฉพาะ ให้ระบุชื่อชิ้นส่วนนั้นสั้น ๆ ก่อนบอกค่า ห้ามเหมารวมว่าเป็นค่าของทุกชิ้นส่วน
12. ให้ยึดข้อมูลหลักอันดับ 1 เป็นคำตอบหลัก ใช้ข้อมูลอันดับอื่นเฉพาะเมื่อสนับสนุนเรื่องเดียวกัน ห้ามนำข้อมูลคนละระบบหรือคนละอาการมารวมกัน
13. พูดถึงสาเหตุเฉพาะเมื่อข้อมูลอ้างอิงระบุความสัมพันธ์ว่าเป็นสาเหตุโดยตรงเท่านั้น ห้ามเปลี่ยนรายการตรวจสอบให้กลายเป็นข้อสรุปว่าชิ้นส่วนเสีย
14. หากคู่มือให้ตรวจหลายจุดก่อนวินิจฉัย ให้เรียงสิ่งที่ต้องตรวจตามลำดับในคู่มือไม่เกิน 4 ข้อ (ใช้เลข 1) 2) 3)... ในประโยคเดียวกัน)
    ห้ามสรุปให้เปลี่ยนอะไหล่ตัวแรกทันที
15. บอกวิธีแก้เฉพาะเมื่อข้อมูลอ้างอิงระบุการแก้ไขสำหรับเงื่อนไขที่ยืนยันแล้วโดยตรง หากยังเป็นเพียงขั้นตอนวินิจฉัย ให้บอกแค่สิ่งที่ต้องตรวจ ไม่ต้องสรุปวิธีแก้เอง
15.1 ข้อความว่า "ตรวจว่าชิ้นส่วนทำงานปกติ" เป็นเพียงขั้นตอนตรวจสอบ ไม่ได้อนุญาตให้สรุปว่าเสียหรือให้เปลี่ยนชิ้นส่วนนั้น
16. ค่ามาตรฐานต้องคงตัวเลข หน่วย ชิ้นส่วน และเงื่อนไขการวัดจากคู่มือ ห้ามนำค่าจากคนละโหมดหรือคนละเงื่อนไขมาเปรียบเทียบกัน
17. หลักฐานที่ยืนยันเป็นเพียงจุดบอกว่าหน้านี้ตรงคำถาม ให้ใช้เนื้อหาเต็มของข้อมูลหลักเพื่อรักษาบริบทและลำดับ แต่ห้ามนำหัวข้อข้างเคียงที่ไม่เกี่ยวข้องมาตอบ
17.1 ถ้าข้อมูลอ้างอิงมีตาราง Trouble/Cause/Remedy ที่แบ่งปัญหาย่อยหลายข้อในอะไหล่ชิ้นเดียวกัน
     (เช่น "1) หมุนไม่ได้ 2) น้ำมันรั่ว 3) ร้อนผิดปกติ") ให้ใช้สาเหตุ/วิธีแก้จากข้อที่ตรงกับอาการ
     ที่ลูกค้าถามเพียงข้อเดียวเท่านั้น ห้ามนำสาเหตุจากปัญหาย่อยข้ออื่นในตารางเดียวกันมารวมตอบ
     แม้จะอยู่ในอะไหล่ชิ้นเดียวกันก็ตาม
18. เมื่อพูดถึงสาเหตุ ให้บอกตรงๆ ห้ามใช้คำว่า "อาจ" หรือ "อาจจะ" หรือคำเลี่ยงความชัดเจนอื่น
19. คำตอบต้องลงท้ายด้วยคำว่า "ครับ" ให้ฟังดูสุภาพเหมือนช่างคุยกับลูกค้า
20. ใช้คำศัพท์ช่างให้คงที่: relief valve = วาล์วระบายแรงดัน, pump proportional valve = วาล์วสัดส่วนปั๊ม และ pump regulator = เรกูเลเตอร์ปั๊ม ห้ามแปล relief valve เป็นลิฟท์วาล์ว
""".strip()

# ต้องตรงกับข้อความในกฎข้อ 8 ของ SYSTEM_PROMPT ด้านบนเป๊ะ ใช้เช็คใน webhook.py ว่า
# ask_llm ตัดสินใจว่าหลักฐานไม่พอ (ทั้งที่ rerank เลือก chunk มาแล้ว) จะได้ไม่แนบรูปอ้างอิง
# ของ chunk นั้นไปกับคำตอบที่บอกว่า "ไม่พบข้อมูล" ซึ่งจะดูขัดแย้งกันเอง
INSUFFICIENT_DATA_MARKER = "ไม่พบข้อมูลเพียงพอในฐานข้อมูล"


def is_insufficient_data_answer(answer: str) -> bool:
    return INSUFFICIENT_DATA_MARKER in answer


_anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
_groq_client = AsyncOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
_SEARCH_QUERY_CACHE_MAX = 512
_MIN_EVIDENCE_CONFIDENCE = 0.55
# เคยลดเหลือ 1 เพื่อเพิ่ม recall แต่พบจริงว่าดึงหน้าที่แค่มีคำซ้ำผิวเผิน (เช่น หน้าติดตั้งปั๊มใหม่
# มาตอบคำถามเรื่องปั๊มไม่สร้างแรงดัน) กลับไป 2 เพื่อกันหลักฐานอ่อนเกินไปหลุดผ่านมาตอบผิดเรื่อง
_SAFE_FALLBACK_MIN_SUPPORT = 2
_MAX_RERANK_SELECTIONS = 3
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
    """ถามกลับเฉพาะกรณีคำเดียวกันสื่อได้ 2 ความหมายที่ต่างกันสุดขั้วจริงๆ (ไม่ใช่แค่มั่นใจน้อย)

    เคยลบพฤติกรรมถามกลับออกไปทั้งหมด (ดู git log "answer technical questions without
    clarification") เพราะการถามกลับแบบกว้างๆ/ให้ LLM ตัดสินใจเองสร้างภาระให้ช่างต้องพิมพ์
    ไปกลับโดยไม่จำเป็น แต่เคสนี้ (น้ำมันรั่ว vs ชิ้นส่วนหลุดทางกายภาพ) เป็นความกำกวมที่ตอบผิด
    ทางได้จริง จึงนำกลับมาเฉพาะรูปแบบนี้แบบ hardcode แคบๆ ไม่ใช้ LLM ตัดสินใจว่าควรถามหรือไม่
    """
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


def extract_error_code(question: str) -> str | None:
    """ดึงรหัส error code ตัวแรกที่เจอในคำถาม (เช่น A015) ให้เป็นตัวพิมพ์ใหญ่
    คืนค่า None ถ้าคำถามไม่มีรูปแบบรหัส error code
    """
    match = _ERROR_CODE_PATTERN.search(question)
    return match.group(0).upper() if match else None


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


# คู่มือมีตารางที่รวมหลายอาการ/หลายปัญหาไว้ในหน้าเดียวกัน 2 รูปแบบหลักที่เจอจริงแล้วตอบผิด:
#   1) ตาราง "อาการ (แถว) x ปัจจัยที่ต้องเช็ค (คอลัมน์)" (หมวด TROUBLESHOOTING BY TROUBLE)
#      ปัจจัยเดียวกัน (เช่น "leak, clogging of fuel system") ซ้ำอยู่หลายแถวที่อาการต่างกันโดย
#      สิ้นเชิง — เจอจริง: "น้ำมันดีเซลย้อนกลับถังเป็นสีดำ" ไม่มีแถวไหนตรงเลย แต่ถูกตอบด้วยปัจจัย
#      ที่ซ้ำในหลายแถว ไปป์ไลน์ OCR ใส่ marker "Row group:" กำกับตารางแบบนี้ไว้แน่นอนทุกตาราง
#   2) ตาราง Trouble/Cause/Remedy ของอะไหล่ชิ้นเดียว ที่แบ่งปัญหาย่อยเป็นข้อ "1) ... 2) ... 3) ..."
#      เจอจริง: ถามว่า "น้ำมันไฮดรอลิกร้อนเกิดจากอะไร" ได้ตารางของ "หน่วยลดกำลัง (reduction unit)"
#      ที่มีปัญหาย่อย 3 ข้อ (หมุนไม่ได้ / น้ำมันรั่ว / ร้อน) มาตอบ แต่โมเดลผสมสาเหตุจากทั้ง 3 ข้อ
#      เข้าด้วยกัน ทั้งที่ลูกค้าถามแค่เรื่อง "ร้อน" ข้อเดียว
_SYMPTOM_MATRIX_TABLE_MARKER = "Row group:"
_NUMBERED_TROUBLE_ITEM_PATTERN = re.compile(r"(?:^|\n)\s*\d+\)\s+\S")


def _is_symptom_matrix_table(content: str) -> bool:
    if _SYMPTOM_MATRIX_TABLE_MARKER in content:
        return True
    return len(_NUMBERED_TROUBLE_ITEM_PATTERN.findall(content)) >= 2


def _select_safe_fallback(
    chunks: list[RetrievedChunk],
    search_queries: list[str] | None,
    allow_top_ranked: bool = False,
) -> RetrievedChunk | None:
    # ตารางอาการ x ปัจจัย ห้ามหลุดผ่าน fallback แบบ lexical overlap อ่อนๆ เด็ดขาด ต้องมี
    # หลักฐานตรงชื่ออาการจริงเท่านั้น (ตรวจใน _parse_rerank_result ผ่าน evidence quote)
    eligible = [chunk for chunk in chunks if not _is_symptom_matrix_table(chunk.content)]
    scored = [
        (_candidate_support_score(chunk, search_queries), index, chunk)
        for index, chunk in enumerate(eligible)
    ]
    if not scored:
        return None
    support, _, chunk = max(scored, key=lambda item: (item[0], -item[1]))
    if support < _SAFE_FALLBACK_MIN_SUPPORT:
        if allow_top_ranked:
            top_chunk = eligible[0] if eligible else None
            if top_chunk and top_chunk.content.strip():
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
        if _is_symptom_matrix_table(chunks[index].content):
            # ตารางอาการ x ปัจจัย: แค่ verbatim quote จริงยังไม่พอ เพราะโมเดลอาจยกข้อความจริง
            # มาจากคนละแถวที่ไม่เกี่ยวกับอาการที่ถามเลยก็ได้ (ปัจจัยซ้ำกันหลายแถว) ต้องเช็คเพิ่มว่า
            # ตัว evidence เองมีคำที่ทับกับคำค้นพอ ไม่ใช่แค่ทับกับเนื้อหาทั้งหน้าแบบกว้างๆ
            evidence_term_support = _candidate_support_score(
                RetrievedChunk(content=evidence), search_queries
            )
            if not evidence_supported or evidence_term_support < _SAFE_FALLBACK_MIN_SUPPORT:
                rejection_reasons.add("matrix_table_quote_required")
                continue
        elif not evidence_supported and semantic_support < _SAFE_FALLBACK_MIN_SUPPORT:
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
        if len(selected) == _MAX_RERANK_SELECTIONS:
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


# หัวข้อคู่มือที่เป็นขั้นตอนติดตั้ง/ประกอบชิ้นส่วนใหม่ล้วนๆ ไม่ใช่การวินิจฉัยอาการเสีย
# ศัพท์ในหน้าแบบนี้ (pump, hydraulic oil, bleed air, torque) มักซ้ำกับหน้าวินิจฉัยจริงเยอะมาก
# จน lexical overlap ทั่วไปแยกไม่ออก จึงต้องกันด้วยชื่อหัวข้อโดยตรง
_PROCEDURE_ONLY_SECTION_TERMS = (
    "installation",
    "assembly",
    "removal and installing",
    "disassembly",
    "reassembly",
)


def _is_procedure_only_heading(heading_area: str) -> bool:
    normalized = heading_area.casefold()
    return any(term in normalized for term in _PROCEDURE_ONLY_SECTION_TERMS)


_PROCEDURE_SEEKING_CATEGORIES = {
    "procedure inspection adjustment",
    "component part number removal installation",
}


def _filter_procedure_only_pages(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """อย่าเอาหน้าติดตั้ง/ประกอบชิ้นส่วนใหม่มาตอบคำถามวินิจฉัยอาการเสีย

    เจอจริงจากบทสนทนา: คำถาม "ปั๊มไม่สร้างแรงดัน" ถูกตอบด้วยหน้า "Installing the pump"
    เพราะศัพท์ซ้ำกันเยอะ (pump, hydraulic oil, bleed air) ทั้งที่คนละเรื่องกันเลย
    ใช้ allowlist แคบๆ แทนการเช็คว่าเป็นคำถามวินิจฉัยหรือไม่ เพราะคำถามจริงมักมีคำวัดค่า
    (เช่น "แรงดัน") ปนอยู่ ทำให้ infer_search_category_hint จัดเป็นหมวดวัดค่าแทนหมวดวินิจฉัย
    ทั้งที่ยังเป็นคำถามอาการเสียอยู่ดี จึงกรองออกเป็นค่าเริ่มต้นเสมอ ยกเว้นคำถามจะขอวิธีติดตั้ง/
    ประกอบ/อะไหล่ตรงๆ (2 หมวดนี้เท่านั้น) และไม่กรองจนเหลือ 0 (กันเคสหน้าเดียวที่เจอเป็นหน้า
    ติดตั้งจริงๆ แต่เป็นหลักฐานที่ดีที่สุดที่มี)
    """
    if infer_search_category_hint(question) in _PROCEDURE_SEEKING_CATEGORIES:
        return chunks
    filtered = [
        chunk for chunk in chunks if not _is_procedure_only_heading(chunk.content[:2000])
    ]
    return filtered if filtered else chunks


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
    chunks = _filter_procedure_only_pages(question, chunks)
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
    # เผื่อโมเดลยังเผลอเลี่ยงคำ ให้ตัดคำเลี่ยงความชัดเจนออกโดยไม่แทรกหัวข้อใดๆ กลับเข้าไปแทน
    answer = re.sub(
        r"อาจ(?:จะ)?(?:มาจาก|เกิดจาก|เป็นเพราะ)\s*",
        "เกิดจาก",
        answer,
    )
    answer = re.sub(r"อาจ(?:จะ)?\s*", "", answer)
    # เผื่อโมเดลยังเผลอใส่หัวข้อ/label แม้กฎจะห้ามไว้แล้ว ให้ตัดออกเงียบๆ ไม่ให้หลุดไปถึงลูกค้า
    answer = re.sub(r"(?:สาเหตุ|ตรวจสอบ|ค่ามาตรฐาน|วิธีแก้)\s*:\s*", "", answer)

    compact_lines = []
    for raw_line in answer.splitlines():
        line = " ".join(raw_line.strip().lstrip("-•* ").split())
        if not line:
            continue

        # Keep a concise ordered diagnostic path instead of silently discarding every
        # check after item 1. The prompt limits the list to four relevant items.
        line = re.sub(r"(?<=:)\s*1[.)]\s*", " 1) ", line)
        # คำตอบปกติตอนนี้เป็นข้อความเดียวไม่มีหัวข้อแยกบรรทัด จึงตัดที่ ~590 ตัวอักษร
        # (ใกล้เพดานรวม 600 ตัวอักษร) แทนเพดานเดิม 220 ที่คิดไว้สำหรับหัวข้อสั้นๆ หลายบรรทัด
        if len(line) > 590:
            cut_at = max(
                line.rfind(".", 0, 591),
                line.rfind(";", 0, 591),
                line.rfind(",", 0, 591),
            )
            if cut_at >= 350:
                line = line[: cut_at + 1]
            else:
                boundaries = [
                    line.find(separator, 350, 591)
                    for separator in (" และ", " หรือ", " ซึ่ง", " โดย", " ทำให้")
                ]
                boundaries = [position for position in boundaries if position >= 350]
                if boundaries:
                    line = line[: min(boundaries)].rstrip(" ,;:-")
                else:
                    cut_at = line.rfind(" ", 0, 588)
                    if cut_at < 350:
                        cut_at = 587
                    line = line[:cut_at].rstrip(" ,;:-")
            # ถ้าตัดจบกลางเครื่องหมายคำพูดที่เปิดค้างไว้ (เช่น กลางคำพูดภาษาอังกฤษที่ยกมา)
            # จะอ่านดูขาดตอน ตัดกลับไปก่อนหน้าเครื่องหมายเปิดนั้นทั้งท่อนแทน
            last_open_quote = line.rfind("“")
            if last_open_quote != -1 and "”" not in line[last_open_quote:]:
                line = line[:last_open_quote].rstrip(" ,;:-–")
        compact_lines.append(line)
        if len(compact_lines) == 4:
            break

    selected_lines = []
    current_length = 0
    for line in compact_lines:
        added_length = len(line) + (1 if selected_lines else 0)
        if selected_lines and current_length + added_length > 600:
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


_GENERIC_CLARIFICATION_FALLBACK = (
    "ไม่พบข้อมูลเพียงพอในฐานข้อมูล กรุณาระบุชิ้นส่วนหรืออาการให้ชัดเจนขึ้นครับ"
)


async def ask_clarifying_question(question: str) -> str:
    """ถามกลับแบบเจาะจงเมื่อระบบกำลังจะตอบว่าไม่พบข้อมูลอยู่แล้วเท่านั้น

    ต่างจาก get_ambiguity_clarification ที่ถามก่อนแม้บางทีจะตอบได้ - ฟังก์ชันนี้เรียกเฉพาะตอน
    "จนตรอก" แล้ว (ไม่มี evidence ผ่าน rerank หรือ ask_llm บอกว่าหลักฐานไม่พอ) จึงไม่มีความเสี่ยง
    ที่จะไปขัดจังหวะคำถามที่ตอบได้ดีอยู่แล้ว ผิดจากกลไกถามกลับกว้างๆ ที่เคยถูกถอดออกไปก่อนหน้า
    """
    try:
        if settings.llm_provider == "claude":
            resp = await _anthropic_client.messages.create(
                model=settings.claude_model,
                max_tokens=120,
                temperature=0,
                system=CLARIFYING_QUESTION_PROMPT,
                messages=[{"role": "user", "content": question}],
            )
            text = resp.content[0].text
        else:
            resp = await _groq_client.chat.completions.create(
                model=settings.groq_model,
                max_completion_tokens=min(settings.llm_max_tokens, 200),
                temperature=0,
                extra_body={"reasoning_effort": "low"},
                messages=[
                    {"role": "system", "content": CLARIFYING_QUESTION_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            text = resp.choices[0].message.content or ""
        text = " ".join(text.split())
        if not text or len(text) > 300:
            return _GENERIC_CLARIFICATION_FALLBACK
        return text
    except Exception:  # noqa: BLE001
        return _GENERIC_CLARIFICATION_FALLBACK
