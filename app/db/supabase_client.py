from supabase import create_client, Client
from app.config import settings

_client: Client | None = None


def get_supabase() -> Client:
    """คืนค่า Supabase client ตัวเดียว ใช้ร่วมกันทั้งแอพ (ไม่สร้าง connection ใหม่ทุก request)"""
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_service_key)
    return _client
