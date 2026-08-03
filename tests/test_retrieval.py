from app.services.claude_service import (
    _cache_search_queries,
    _clean_answer,
    _question_cache_key,
    _search_query_cache,
    _parse_search_queries,
)
from app.services.retrieval_service import _rank_results


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
    answer = (
        "ค่ามาตรฐาน 27.5 Ω [แหล่ง 17-43]\n"
        "ตรวจที่ขา 1-5 [source: page 17-43] (อ้างอิงจากคู่มือ)"
    )
    assert _clean_answer(answer) == "ค่ามาตรฐาน 27.5 Ω\nตรวจที่ขา 1-5"


def test_question_cache_normalizes_case_and_whitespace():
    assert _question_cache_key("  SK200-8   Pump  ") == "sk200-8 pump"


def test_search_query_cache_reuses_normalized_question():
    _search_query_cache.clear()
    key = _question_cache_key("SK200-8 ปั๊มช้า")
    _cache_search_queries(key, ["SK200-8 pump slow"])
    assert list(_search_query_cache[key]) == ["SK200-8 pump slow"]


def test_rank_results_is_stable_and_rewards_cross_search_matches():
    exact_only = {"id": "a", "content": "exact", "search_score": 0.9}
    shared = {"id": "b", "content": "shared", "search_score": 0.8}
    vector_only = {"id": "c", "content": "vector", "similarity": 0.95}
    ranked = _rank_results(
        [
            (3.0, [exact_only, shared]),
            (1.5, [shared]),
            (2.0, [vector_only, shared]),
        ],
        limit=3,
    )
    assert [row["id"] for row in ranked] == ["b", "a", "c"]


def test_rank_results_breaks_equal_scores_by_stable_id():
    ranked = _rank_results(
        [(1.0, [{"id": "b"}, {"id": "a"}])],
        limit=2,
    )
    assert [row["id"] for row in ranked] == ["a", "b"]
