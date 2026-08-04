# CLAUDE.md — LINE Repair Bot (บริษัท TIS)

## โปรเจกต์นี้คืออะไร
บอทตอบคำถามซ่อม/บำรุงรักษารถขุด KOBELCO SK200-8 ผ่าน LINE Official Account
ให้ช่างเทคนิคของบริษัท TIS ("ช่างเต้") ถามอาการเสีย/error code/ค่ามาตรฐาน แล้วบอทค้นคำตอบจากคู่มือซ่อมจริง
(SK200-8.pdf) ที่แปลงเป็นข้อมูลไว้ใน Supabase แล้วส่งให้ LLM (Groq หรือ Claude สลับได้) เรียบเรียงเป็นคำตอบภาษาไทย

## สถาปัตยกรรมคร่าวๆ
LINE webhook → แยกเจตนา (ทักทาย/ถามแค่รุ่นรถ/ถาม error code ตรงๆ/คำถามเทคนิคทั่วไป)
→ ค้นข้อมูล (ตาราง error code ที่ยืนยันแล้ว ก่อนเสมอถ้ามีรหัส / ไม่งั้น vector+full-text search) → LLM คัดกรองหลักฐาน (rerank)
→ LLM เรียบเรียงคำตอบภาษาไทย → ตอบกลับ LINE พร้อมรูปอ้างอิงถ้ามี

รายละเอียด flow, schema, กติกาการค้นหาเต็มๆ อยู่ใน SPEC.md

## กฎเหล็ก (ห้ามฝ่าฝืน)
- **ห้ามตอบจากความรู้ทั่วไปของ LLM เด็ดขาด** ต้องตอบจากข้อมูลอ้างอิงที่ดึงมาจาก Supabase เท่านั้น
  (กฎนี้ล็อกไว้แน่นแล้วใน `SYSTEM_PROMPT`/`RERANK_PROMPT` ของ `claude_service.py` — ห้ามผ่อนเงื่อนไขนี้โดยไม่ถามก่อน)
- ถ้าหลักฐานไม่พอ ต้องตอบว่าไม่พบข้อมูลเพียงพอ ห้ามเดา ห้ามแต่งค่ามาตรฐาน/ขั้นตอนซ่อมขึ้นมาเอง
- คำถามที่ระบุ error code ชัดเจน ต้องเช็ค `error_codes_sk2008` (ตารางที่ verified แล้ว) ก่อนเสมอ
  ถ้าไม่เจอในตารางนี้ค่อย fallback ไปค้น `documents_gemini` (ดู SPEC.md หัวข้อ retrieval)
- ห้ามลบ/แก้ embedding โดยไม่สร้างใหม่ให้ตรงกับ content ที่แก้ (มิติ/โมเดลต้องตรงกับตอน insert ครั้งแรกเป๊ะ)
- คำตอบที่ส่งกลับ LINE ต้องเป็นภาษาไทย กระชับ ไม่เกิน 350 ตัวอักษร ไม่มีเลขหน้า/แหล่งอ้างอิงปนในเนื้อความ
  (ระบบเติมบรรทัด "รูปอ้างอิง"/คำเตือนให้เองแล้วในชั้น `webhook.py`)
- ห้าม commit ค่า secret ใดๆ ใน `.env` ขึ้น git

## ศัพท์ที่ใช้ในโปรเจกต์
- **chunk** = 1 แถวใน `documents_gemini` (เนื้อหาคู่มือ 1 ส่วน + embedding)
- **rerank** = ขั้นให้ LLM เป็น "evidence gate" คัดว่า chunk ไหนตรงคำถามจริงก่อนค่อยให้เรียบเรียงคำตอบ
- **verified error code** = แถวใน `error_codes_sk2008` ที่ผ่าน QA แล้ว (`verified=true`, `review_status=approved`)
  ความแม่นยำสูงกว่า chunk OCR ดิบใน `documents_gemini` มาก
- **document_id_exact / CURRENT_DOCUMENT_ID** = ตัวล็อกเวอร์ชันข้อมูล ปัจจุบันคือ `kobelco-sk2008-repair-ocr-v4`
  ห้ามลืมอัปเดตทุกจุดที่ hardcode ค่านี้ถ้าจะออกข้อมูลชุดใหม่ (v5, ...)

## กติกาเทคนิค
- Stack: FastAPI + Supabase (Postgres/pgvector) + Groq หรือ Claude (สลับด้วย `LLM_PROVIDER` ใน `.env`)
- Embedding ต้องใช้ Gemini เสมอ (`gemini-embedding-001`, มิติ 3072) ไม่ว่าจะสลับ LLM/vision provider เป็นอะไรก็ตาม
- ห้ามแก้ prompt ใน `claude_service.py` (SYSTEM_PROMPT/RERANK_PROMPT/SEARCH_QUERY_PROMPT) แบบเดาสุ่ม
  แต่ละกฎในนั้นมักแก้มาจาก production bug จริง (ดู `git log` ของไฟล์นี้ก่อนแก้ เพื่อไม่ย้อนกลับไปแก้บั๊กเดิม)
- มี `tests/` คู่กับทุกการเปลี่ยนแปลง logic ใน `claude_service.py`/`retrieval_service.py`/`webhook.py` — รัน `pytest` ก่อนบอกว่าเสร็จเสมอ
  (ใช้ `.venv/Scripts/python.exe -m pytest` เพราะ python บน PATH หลักอาจเป็นคนละตัวที่ไม่มี dependency ครบ)
- สคริปต์เสริม (เช่น `scripts/fix_table_ocr.py`) แก้ข้อมูลใน Supabase โดยตรง — เป็นการเปลี่ยนแปลง production data
  ต้องแจ้งผู้ใช้ก่อนรันจริง (ไม่ใช่ dry-run) ทุกครั้ง

## วิธีทำงานกับฉัน
- ก่อนเชื่อมข้อมูลตารางใหม่ๆ ใน Supabase เข้า retrieval flow ให้เช็คก่อนว่าโค้ดปัจจุบันอ้างอิงตารางนั้นอยู่แล้วหรือยัง
  (เจอมาแล้วว่ามีตารางที่มีข้อมูลจริงแต่ไม่ได้ต่อเข้าโค้ดเลย เช่น `error_codes_sk2008` ก่อนหน้านี้)
- เวลาทดสอบที่ต้องยิง Gemini API จริง ให้ระวัง rate limit ฝั่ง free tier (เจอปัญหานี้บ่อย) — ทดสอบเท่าที่จำเป็นพอ
- ตอบและอธิบายเป็นภาษาไทย
