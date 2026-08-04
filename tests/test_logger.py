"""เทส log_conversation: ต้องบันทึกทั้ง user/assistant ลง Supabase และไม่พังถ้าเขียนไม่สำเร็จ"""
from unittest.mock import MagicMock, patch

from app.utils.logger import log_conversation


def test_log_conversation_persists_user_and_assistant_rows():
    fake_supabase = MagicMock()
    with patch("app.utils.logger.get_supabase", return_value=fake_supabase):
        log_conversation("Uabc123", "เครื่องร้อนทำไงดี", "ตรวจน้ำมันเครื่องครับ")

    fake_supabase.table.assert_called_once_with("line_conversation_messages")
    inserted_rows = fake_supabase.table.return_value.insert.call_args[0][0]
    assert inserted_rows == [
        {"line_user_id": "Uabc123", "role": "user", "message_text": "เครื่องร้อนทำไงดี"},
        {"line_user_id": "Uabc123", "role": "assistant", "message_text": "ตรวจน้ำมันเครื่องครับ"},
    ]
    fake_supabase.table.return_value.insert.return_value.execute.assert_called_once()


def test_log_conversation_does_not_raise_when_supabase_write_fails():
    fake_supabase = MagicMock()
    fake_supabase.table.return_value.insert.return_value.execute.side_effect = RuntimeError("boom")
    with patch("app.utils.logger.get_supabase", return_value=fake_supabase):
        log_conversation("Uabc123", "คำถาม", "คำตอบ")
