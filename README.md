# LINE Repair Bot (FastAPI)

บอทตอบคำถามซ่อมเครื่องจักรผ่าน LINE OA — ดึงข้อมูลจากคู่มือ/error code ที่เก็บใน Supabase (pgvector)
แล้วให้ LLM (Groq หรือ Claude สลับได้) สรุปคำตอบ

## เริ่มต้นใช้งาน

1. ติดตั้ง dependencies
   ```
   pip install -r requirements.txt
   ```

2. คัดลอก `.env.example` เป็น `.env` แล้วใส่ค่าจริง (LINE token, Supabase, Groq/Anthropic API key)

3. รันเซิร์ฟเวอร์
   ```
   uvicorn app.main:app --reload
   ```

4. ตั้ง webhook URL ใน LINE Developer Console ให้ชี้มาที่ `https://<your-domain>/webhook`

## สลับ LLM provider

ตั้งค่า `LLM_PROVIDER=groq` หรือ `LLM_PROVIDER=claude` ใน `.env` — ไม่ต้องแก้โค้ด

## โครงสร้างโปรเจกต์

```
app/
├── main.py                 FastAPI entrypoint
├── config.py                โหลด env vars
├── routers/webhook.py       รับ webhook จาก LINE
├── services/
│   ├── line_service.py      เรียก LINE Messaging API
│   ├── embedding_service.py สร้าง embedding คำถาม
│   ├── retrieval_service.py query Supabase pgvector
│   └── claude_service.py    เรียก LLM (Groq/Claude) ตอบคำถาม
├── models/schemas.py        Pydantic models
├── db/supabase_client.py    Supabase client
└── utils/logger.py          log บทสนทนา
```

## หมายเหตุ

- ต้องมี Supabase RPC function ชื่อ `match_chunks(query_embedding, match_count)` อยู่แล้ว (ใช้ของเดิมจาก pipeline OCR+chunk)
- ต้องมี table `conversation_logs` สำหรับเก็บ log บทสนทนา (สร้างเองถ้ายังไม่มี)
- Embedding model ต้องตรงกับตัวที่ใช้ตอน insert ข้อมูลเข้า Supabase ครั้งแรก
