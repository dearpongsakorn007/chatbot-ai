from app.services.claude_service import _clean_answer, _parse_search_queries


def test_parse_search_queries():
    queries = _parse_search_queries(
        "solenoid coil resistance\nsolenoid resistance specification\nelectrical coil test"
    )
    assert queries == [
        "solenoid coil resistance",
        "solenoid resistance specification",
        "electrical coil test",
    ]


def test_parse_search_queries_removes_list_markers_and_duplicates():
    queries = _parse_search_queries(
        "1. boom hydraulic pressure\n- boom hydraulic pressure\n• boom slow operation"
    )
    assert queries == ["boom hydraulic pressure", "boom slow operation"]


def test_parse_search_queries_rejects_empty_and_long_lines():
    assert _parse_search_queries("\n" + ("x" * 121)) == []


def test_clean_answer_removes_model_generated_source_labels():
    answer = "ค่ามาตรฐาน 27.5 Ω [แหล่ง 17-43]\nตรวจที่ขา 1-5 [source: page 17-43]"
    assert _clean_answer(answer) == "ค่ามาตรฐาน 27.5 Ω\nตรวจที่ขา 1-5"
