"""เทส conversation_state_service: ผูกคำตอบสั้นๆ ของลูกค้ากลับเข้ากับคำถามเดิมที่บอทถามกลับไป"""
from unittest.mock import MagicMock, patch

from app.services.conversation_state_service import (
    clear_pending_clarification,
    get_pending_clarification,
    save_pending_clarification,
)


def test_get_pending_clarification_returns_question_and_round_count_when_found():
    fake_supabase = MagicMock()
    fake_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.gt.return_value.limit.return_value.execute.return_value.data = [
        {
            "question_detail": "เดิมถามเรื่องน้ำมันไฮดรอลิก",
            "context": {"clarification_rounds": 1},
            "expires_at": "2099-01-01T00:00:00+00:00",
        }
    ]
    with patch(
        "app.services.conversation_state_service.get_supabase",
        return_value=fake_supabase,
    ):
        result = get_pending_clarification("Uabc123")
    assert result == ("เดิมถามเรื่องน้ำมันไฮดรอลิก", 1)


def test_get_pending_clarification_defaults_rounds_to_zero_when_missing():
    fake_supabase = MagicMock()
    fake_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.gt.return_value.limit.return_value.execute.return_value.data = [
        {"question_detail": "เดิมถามเรื่องน้ำมันไฮดรอลิก", "context": {}, "expires_at": "2099-01-01T00:00:00+00:00"}
    ]
    with patch(
        "app.services.conversation_state_service.get_supabase",
        return_value=fake_supabase,
    ):
        result = get_pending_clarification("Uabc123")
    assert result == ("เดิมถามเรื่องน้ำมันไฮดรอลิก", 0)


def test_get_pending_clarification_returns_none_when_no_row():
    fake_supabase = MagicMock()
    fake_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.gt.return_value.limit.return_value.execute.return_value.data = []
    with patch(
        "app.services.conversation_state_service.get_supabase",
        return_value=fake_supabase,
    ):
        assert get_pending_clarification("Uabc123") is None


def test_get_pending_clarification_returns_none_on_error():
    fake_supabase = MagicMock()
    fake_supabase.table.return_value.select.side_effect = RuntimeError("boom")
    with patch(
        "app.services.conversation_state_service.get_supabase",
        return_value=fake_supabase,
    ):
        assert get_pending_clarification("Uabc123") is None


def test_save_pending_clarification_upserts_awaiting_status_and_round_count():
    fake_supabase = MagicMock()
    with patch(
        "app.services.conversation_state_service.get_supabase",
        return_value=fake_supabase,
    ):
        save_pending_clarification("Uabc123", "คำถามเดิม", rounds=2)

    upsert_call = fake_supabase.table.return_value.upsert
    payload = upsert_call.call_args[0][0]
    assert payload["line_user_id"] == "Uabc123"
    assert payload["question_detail"] == "คำถามเดิม"
    assert payload["status"] == "collecting"
    assert payload["context"] == {"clarification_rounds": 2}
    assert upsert_call.call_args.kwargs["on_conflict"] == "line_user_id"


def test_clear_pending_clarification_deletes_the_row():
    fake_supabase = MagicMock()
    with patch(
        "app.services.conversation_state_service.get_supabase",
        return_value=fake_supabase,
    ):
        clear_pending_clarification("Uabc123")
    fake_supabase.table.return_value.delete.return_value.eq.assert_called_once_with(
        "line_user_id", "Uabc123"
    )
