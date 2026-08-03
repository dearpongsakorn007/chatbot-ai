import json

from app.models.schemas import RetrievedChunk
from app.services.claude_service import (
    _cache_search_queries,
    _clean_answer,
    get_ambiguity_clarification,
    _question_cache_key,
    _search_query_cache,
    _parse_rerank_result,
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


def test_ambiguous_injector_wording_requests_clarification():
    question = "SK200-8 หัวฉีดน้ำมันดันออกหลุดออกมาเกิดจากอะไร"
    clarification = get_ambiguity_clarification(question)
    assert clarification is not None
    assert "หลุดออกจากฝาสูบ" in clarification
    assert "น้ำมันรั่ว" in clarification


def test_explicit_physical_or_leak_wording_skips_clarification():
    assert get_ambiguity_clarification("หัวฉีดน้ำมันหลุดออกจากฝาสูบ") is None
    assert get_ambiguity_clarification("น้ำมันรั่วบริเวณหัวฉีด") is None


def test_parse_rerank_result_requires_verbatim_high_confidence_evidence():
    chunks = [
        RetrievedChunk(content="Broken pipe causes fuel leak at injector connection", reference="17-39")
    ]
    raw = json.dumps(
        {
            "status": "answer",
            "clarification": "",
            "selections": [
                {
                    "index": 1,
                    "confidence": 0.91,
                    "evidence": "Broken pipe causes fuel leak at injector connection",
                }
            ],
        }
    )
    result = _parse_rerank_result(raw, chunks)
    assert result.clarification is None
    assert [chunk.reference for chunk in result.chunks] == ["17-39"]
    assert result.chunks[0].content == "Broken pipe causes fuel leak at injector connection"


def test_parse_rerank_result_rejects_low_confidence_or_invented_evidence():
    chunks = [RetrievedChunk(content="Replace injector", reference="16-3")]
    low_confidence = (
        '{"status":"answer","selections":'
        '[{"index":1,"confidence":0.4,"evidence":"Replace injector due to physical detachment"}]}'
    )
    invented = (
        '{"status":"answer","selections":'
        '[{"index":1,"confidence":0.95,"evidence":"Injector clamp became loose and detached"}]}'
    )
    assert _parse_rerank_result(low_confidence, chunks).chunks == []
    assert _parse_rerank_result(invented, chunks).chunks == []


def test_parse_rerank_result_returns_clarification_without_evidence():
    chunks = [RetrievedChunk(content="Replace injector", reference="16-3")]
    raw = json.dumps(
        {
            "status": "clarify",
            "clarification": "หมายถึงตัวหัวฉีดหลุดจากฝาสูบ หรือน้ำมันรั่วบริเวณหัวฉีดครับ?",
            "selections": [],
        },
        ensure_ascii=False,
    )
    result = _parse_rerank_result(raw, chunks)
    assert result.chunks == []
    assert "หัวฉีดหลุด" in result.clarification


def test_clean_answer_removes_model_generated_source_labels():
    answer = (
        "ค่ามาตรฐาน 27.5 Ω [แหล่ง 17-43]\n"
        "ตรวจที่ขา 1-5 [source: page 17-43] (อ้างอิงจากคู่มือ)"
    )
    assert _clean_answer(answer) == "ค่ามาตรฐาน 27.5 Ω\nตรวจที่ขา 1-5"


def test_clean_answer_replaces_uncertain_cause_wording():
    answer = "สาเหตุอาจมาจากแรงดันปั๊มต่ำ ซึ่งอาจจะทำให้เครื่องช้า"
    assert _clean_answer(answer) == "สาเหตุ: แรงดันปั๊มต่ำ ซึ่งทำให้เครื่องช้า"


def test_clean_answer_collapses_duplicate_causes_and_enforces_length():
    answer = (
        "สาเหตุ: แรงดันปั๊มต่ำ สาเหตุ: วาล์วผิดปกติ\n"
        "ตรวจสอบ: ตรวจแรงดันปั๊ม และตรวจฟิลเตอร์\n"
        "วิธีแก้: 1) ปรับแรงดันปั๊ม 2) เปลี่ยนวาล์ว"
    )
    cleaned = _clean_answer(answer)
    assert cleaned.count("สาเหตุ:") == 1
    assert len(cleaned) <= 350
    assert "ตรวจฟิลเตอร์" not in cleaned
    assert "เปลี่ยนวาล์ว" not in cleaned
    assert "…" not in cleaned


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
