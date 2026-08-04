# TASKS.md — งานค้าง/กำลังทำ (LINE Repair Bot)

หมายเหตุ: ไม่ใช่ลำดับคอร์สเรียนแบบตายตัว — อัปเดตไฟล์นี้เมื่อสถานะงานเปลี่ยนจริง (เช่น OCR fix ทำไปกี่แถวแล้ว)

## กำลังทำอยู่

### 1. แก้ OCR ตารางเพี้ยนใน `documents_gemini` (`scripts/fix_table_ocr.py`)
- สถานะ: **67 / 213 แถว** สำเร็จแล้ว (เช็คล่าสุด) รันด้วย `VISION_MODEL=gemini-flash-latest` เพราะ `gemini-2.5-flash`
  ใช้ไม่ได้แล้ว (deprecated) และ Groq ไม่มีโมเดล vision ให้ใช้เลย
- รันต่อได้ตรงๆ (idempotent, resume เองจาก `fix_table_ocr_progress.jsonl`):
  ```
  VISION_MODEL=gemini-flash-latest python scripts/fix_table_ocr.py --pdf "SK200-8.pdf"
  ```
- แถวที่ตกไปแล้วต้องดูเป็นกรณีพิเศษ (ไม่ใช่แค่รันซ้ำเฉยๆ):
  - id 6204 — Gemini ปฏิเสธด้วย `finishReason=RECITATION` ต้องหาสาเหตุว่าเนื้อหาหน้านั้นไปชนอะไร
    (อาจต้องลด dpi/quality หรือ crop เฉพาะส่วนตาราง แล้วลองใหม่)
  - แถวที่โดน 429 ประปราย — ลองรันซ้ำได้เลย ปกติผ่านในรอบถัดไป
- เช็ค quota ก่อนรันรอบใหญ่เสมอ: ยิง `generateContent` ตรงๆ 1 ครั้งดูก่อนว่า 200 หรือ 429
  (เจอปัญหาตลอดบ่ายนี้ว่าเปลี่ยน API key ในโปรเจกต์เดิมไม่ช่วย เพราะ quota ผูกกับ Google Cloud project ไม่ใช่ตัวคีย์)

## งานที่เพิ่งปิดไปแล้ว
- ✅ เชื่อม `error_codes_sk2008` (ตาราง error code ที่ verified แล้ว) เข้า retrieval flow —
  คำถามที่ระบุ error code ตรงๆ จะตอบจากตารางนี้ก่อนเสมอ ไม่ต้องพึ่ง chunk OCR ที่ยังมีปัญหา
  (`retrieval_service.lookup_error_code`, `claude_service.extract_error_code`, ต่อสายใน `webhook._handle_message`)
  มี smoke test ผ่านจริงแล้ว (ดู commit `1359db1`)

## รอตัดสินใจ / ยังไม่ได้เริ่ม
- **`manual_knowledge_sk2008`** (3044 แถว) ยังไม่ได้เชื่อมกับโค้ดเลย ต้องตรวจก่อนว่าข้อมูลซ้ำ/เสริมกับ
  `documents_gemini` ยังไง แล้วค่อยตัดสินใจว่าจะใช้แทน ใช้เสริม หรือไม่ต้องใช้เลย
- **`technician_tickets`, `line_conversation_state`, `line_conversation_messages`** — ทั้ง 3 ตารางว่างเปล่า
  ไม่มี logic อ่าน/เขียนเลย เดาจากชื่อว่าน่าจะเป็นของฟีเจอร์ที่ยังไม่ได้สร้าง (escalate ให้ช่างจริง / เก็บ session /
  เก็บประวัติแชท) ต้องถามเจ้าของโปรเจกต์ว่าตั้งใจสร้างฟีเจอร์ไหนก่อนเริ่มเขียนโค้ด
- CLAUDE.md/SPEC.md/TASKS.md เพิ่งเขียนใหม่รอบนี้ (ของเดิมเป็นเอกสารร้านกาแฟที่ไม่เกี่ยวกับโปรเจกต์นี้เลย
  ถูกลบทิ้งไปแล้วแต่ยังไม่ได้ commit ตอนที่เขียนไฟล์นี้) — ควร commit พร้อมกันเป็นการอัปเดตเอกสารรอบนี้
