# SPEC.md — LINE Repair Bot

## Flow เต็ม (1 ข้อความเข้ามาแล้วเกิดอะไรบ้าง)
`app/routers/webhook.py` → `_handle_message()`:
1. เช็ค signature ของ LINE (`_verify_signature`) — ปฏิเสธถ้าไม่ตรง (401)
2. ถ้าข้อความเป็นคำทักทาย/เปิดบทสนทนา (`_is_conversation_opener`) → ตอบ `OPENING_RESPONSE` ทันที ไม่ค้นข้อมูล
3. ถ้าข้อความมีแค่ชื่อรุ่นรถ (`_model_followup_response`, regex `MODEL_ONLY_PATTERN`) → ถามกลับว่าจะสอบถามเรื่องอะไร ไม่ค้นข้อมูล
4. ถ้าคำถามมีรูปแบบ error code (`extract_error_code`, regex `[A-Za-z]\d{3,4}`) → เช็ค `error_codes_sk2008` ก่อนเสมอ
   (`lookup_error_code`) ถ้าเจอแถวที่ `verified=true` และ `model=SK200-8` → ตอบจากแถวนั้นตรงๆ ผ่าน `ask_llm` แล้วจบเลย
   ข้าม rewrite/vector search/rerank ทั้งหมด (เร็วกว่าและแม่นกว่าเพราะข้อมูลผ่าน QA แล้ว)
5. ถ้าไม่เข้าเงื่อนไขข้อ 4 (ไม่มีโค้ด หรือมีโค้ดแต่ไม่พบในตารางที่ verified) → เข้า pipeline หลัก:
   a. `rewrite_search_queries` — ให้ LLM แปลคำถามเป็นคำค้นภาษาอังกฤษ 3 แบบ (เจาะจง → กว้างขึ้นเรื่อยๆ)
   b. `get_embedding` — ทำ embedding ของคำถาม (+ คำค้นแรก + category hint) ด้วย Gemini
   c. `infer_content_type_filter` — ถ้าคำถามส่อว่าเป็น error code จำกัด content_type ให้แคบลง
   d. `retrieve_chunks` — ดึง chunk จาก `documents_gemini` ด้วย vector search (`match_documents` RPC)
      ผสมกับ full-text search (`search_sk2008_fulltext` RPC) ต่อคำค้นแต่ละแบบ แล้วรวมคะแนนแบบ weighted RRF
   e. `rerank_chunks` — ให้ LLM เป็น evidence gate: กรอง scope mismatch ก่อน (`_filter_scope_mismatches`)
      แล้วให้ LLM เลือกอย่างมาก 2 candidate ที่มีหลักฐานตรงคำถามจริง (verbatim quote หรือ lexical support สำรอง)
   f. ถ้าไม่มี chunk ที่ผ่าน rerank → ตอบ "ไม่พบหลักฐานที่ตรงกับอาการ..." ทันที ไม่เดา
6. `ask_llm` — ประกอบ context จาก chunk ที่ผ่านแล้ว ส่งให้ LLM (`SYSTEM_PROMPT`) เรียบเรียงคำตอบภาษาไทย
   ตามกฎ 350 ตัวอักษร/หัวข้อที่อนุญาต/ห้ามใส่เลขหน้า (ระบบเติมเองทีหลัง)
7. `_prepare_reply` — เติมคำทักทาย + คำตอบ + รูปอ้างอิงของ candidate อันดับ 1 เท่านั้น (ถ้ามี image_url) + คำเตือนท้ายบรรทัด
8. `reply_message` (LINE Messaging API) — ส่งกลับ พร้อม fallback เป็น text-only ถ้าส่งรูปไม่ผ่าน

## Supabase schema (ตารางที่มีจริงตอนนี้)

| ตาราง | สถานะ | ใช้งานยังไง |
|---|---|---|
| `documents_gemini` | ใช้งานจริง (1079 แถว) | RAG chunks + embedding หลักของบอท เวอร์ชันข้อมูลปัจจุบัน `document_id = kobelco-sk2008-repair-ocr-v4` มี `content_type` หลายแบบ (เช่น table, error_code_detail, ...) |
| `error_codes_sk2008` | ใช้งานจริง (100 แถว) | ตาราง error code ที่ผ่าน QA แล้ว (`verified`, `review_status=approved`) เชื่อมเข้า retrieval flow แล้ว (`retrieval_service.lookup_error_code`) เป็นลำดับแรกเสมอเมื่อคำถามระบุโค้ด |
| `manual_knowledge_sk2008` | **ยังไม่ได้เชื่อมกับโค้ด** (3044 แถว) | โครงสร้างข้อมูลคู่มือแบบละเอียด (มี `record_type`, `knowledge_types`, `section_code` ฯลฯ) มาจากไฟล์ `manual_knowledge_sk2008 .json` ที่ root — ยังไม่มีจุดไหนในแอพ query ตารางนี้ ต้องตรวจสอบก่อนว่าซ้ำ/เสริมกับ `documents_gemini` ยังไง |
| `technician_tickets` | **ว่าง ยังไม่ใช้งาน** (0 แถว) | เดาจากชื่อว่าน่าจะไว้ escalate เคสที่บอทตอบไม่ได้ให้ช่างจริง — ยังไม่มี logic ไหนเขียน/อ่านตารางนี้ |
| `line_conversation_state` | **ว่าง ยังไม่ใช้งาน** (0 แถว) | เดาจากชื่อว่าน่าจะไว้เก็บ session/context ต่อผู้ใช้ (เช่น กำลังคุยเรื่องรุ่นไหนอยู่) — ปัจจุบันบอทไม่มี state ข้ามข้อความเลย ทุกข้อความประมวลผลแบบ stateless |
| `line_conversation_messages` | **ว่าง ยังไม่ใช้งาน** (0 แถว) | เดาจากชื่อว่าน่าจะไว้เก็บประวัติแชท — ปัจจุบัน `log_conversation()` ใน `utils/logger.py` แค่ log ลง console (hash user id, ไม่เก็บเนื้อความ) ไม่ได้เขียนลงตารางนี้จริง |

RPC functions ที่ใช้จริง (`match_documents`, `search_sk2008_fulltext`) สร้างไว้ใน Supabase โดยตรง ไม่มีไฟล์ migration
เก็บไว้ในโปรเจกต์ (ยังไม่มีโฟลเดอร์ `migrations/`) — ถ้าจะแก้ signature ต้องเช็คจาก Supabase SQL editor ก่อน

## content_type ที่ระบบรู้จัก (`documents_gemini`)
ใช้กรองตอน error-code path (`ERROR_CODE_CONTENT_TYPES` ใน `claude_service.py`):
`error_code`, `error_code_detail`, `error_code_table`, `error_code_index`
นอกนั้นเป็น content ทั่วไป (procedure/table/narrative ฯลฯ) ไม่ได้ hard filter แต่มี category hint ส่งช่วย vector search
(`infer_search_category_hint`: measurement / symptom / procedure / part)

## OCR quality — งานที่กำลังแก้อยู่ (`scripts/fix_table_ocr.py`)
`documents_gemini` แถวที่ `content_type = table` (213 แถวทั้งหมด) มีปัญหา OCR ตารางเพี้ยนจากการแปลง PDF ครั้งแรก
สคริปต์นี้เรนเดอร์หน้า PDF จริงแล้วให้ Gemini vision อ่านใหม่ คงส่วนหัว citation (`Policy: ANSWERABLE MANUAL CONTENT`)
เดิมไว้ แล้วสร้าง embedding ใหม่ให้ตรงกับ content ที่แก้ (ห้ามข้ามขั้นตอนนี้ ไม่งั้น vector เทียบกับ content เดิมไม่ตรง)
- Resumable: เช็คจาก `scripts/fix_table_ocr_progress.jsonl` (skip แถวที่ `ok: true` แล้ว)
- ต้องใช้ Gemini API key ที่มี quota พอ — ระวัง free tier rate limit (เจอปัญหานี้บ่อยมาก ทั้งแบบ RPM และแบบ RPD ต่อโมเดล)
- Vision model ที่ใช้ได้จริงตอนนี้: `gemini-flash-latest` (alias ที่ Google เลื่อนชี้โมเดล flash รุ่นล่าสุดเสมอ)
  `gemini-2.5-flash` ถูก Google เลิกให้ใช้กับผู้ใช้ใหม่แล้ว (`404 no longer available to new users`)
- Groq **ใช้แทนไม่ได้** สำหรับงานอ่านรูป — บัญชี Groq ที่ผูกอยู่ไม่มีโมเดล vision เลย (เช็คจาก `/v1/models` แล้ว)

## กติกาคำตอบ (สรุปจาก `SYSTEM_PROMPT` ใน `claude_service.py`)
- ใช้เฉพาะข้อมูลอ้างอิงที่ส่งให้ ห้ามเดา
- หัวข้อที่อนุญาต: `สาเหตุ:` `ตรวจสอบ:` `ค่ามาตรฐาน:` `วิธีแก้:` — ใช้เฉพาะหัวข้อที่มีข้อมูลจริงรองรับ
- ยึดข้อมูลอันดับ 1 เป็นหลัก ห้ามผสมข้อมูลคนละระบบ/คนละอาการ
- ไม่เกิน 350 ตัวอักษร ทุกบรรทัดลงท้าย "ครับ" ห้ามมีเลขหน้า/[แหล่งที่มา]/คำว่า "อาจจะ"
- ถ้าหลักฐานไม่พอ: "ไม่พบข้อมูลเพียงพอในฐานข้อมูล กรุณาระบุรุ่นเครื่องหรือ Error Code เพิ่มเติม"
