import asyncio
import json
from unittest.mock import patch

from app.models.schemas import RetrievedChunk
from app.services.claude_service import (
    _cache_search_queries,
    _build_context,
    _clean_answer,
    _candidate_support_score,
    _filter_procedure_only_pages,
    _filter_scope_mismatches,
    _select_safe_fallback,
    ask_clarifying_question,
    get_ambiguity_clarification,
    infer_search_category_hint,
    is_insufficient_data_answer,
    _question_cache_key,
    _search_query_cache,
    _parse_rerank_result,
    _parse_search_queries,
)
from app.services.retrieval_service import _build_page_image_urls, _rank_results


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


def test_parse_search_queries_rejects_model_only_generic_queries():
    assert _parse_search_queries(
        "SK200-8 DIS 3\nSK200-8 diagnostic\nSK200-8 service manual"
    ) == []


def test_parse_search_queries_keeps_exact_error_code_and_useful_terms():
    assert _parse_search_queries(
        "P0217\npump pressure\nslow hydraulic operation"
    ) == ["P0217", "pump pressure", "slow hydraulic operation"]


def test_parse_search_queries_removes_display_position_and_normalizes_pump_index():
    assert _parse_search_queries(
        "DIS 3 pump1\npump1 pressure low\ndisplay 2 service manual"
    ) == ["P1 pump pressure", "P1 pump pressure low"]


def test_ambiguous_injector_wording_now_asks_for_clarification():
    # คำเดียวกันสื่อได้ทั้งหัวฉีดหลุดทางกายภาพ (แก้คนละจุดจากที่รั่ว) — เป็นความกำกวมที่ยอมรับ
    # ให้ถามกลับได้ ต่างจาก "มั่นใจน้อย" ทั่วไปที่ยังตอบภายในขอบเขตที่มีหลักฐานรองรับตามเดิม
    question = "SK200-8 หัวฉีดน้ำมันดันออกหลุดออกมาเกิดจากอะไร"
    assert get_ambiguity_clarification(question) == (
        "หมายถึงตัวหัวฉีดหลุดออกจากฝาสูบ หรือน้ำมันรั่วออกบริเวณหัวฉีดหรือท่อครับ?"
    )


def test_explicit_physical_or_leak_wording_skips_clarification():
    assert get_ambiguity_clarification("หัวฉีดน้ำมันหลุดออกจากฝาสูบ") is None
    assert get_ambiguity_clarification("น้ำมันรั่วบริเวณหัวฉีด") is None


def test_parse_rerank_result_requires_verbatim_high_confidence_evidence():
    chunks = [
        RetrievedChunk(
            content=(
                "Broken pipe causes fuel leak at injector connection. "
                "Check the pipe connection and then replace the pipe if damaged."
            ),
            reference="17-39",
        )
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
    assert "Check the pipe connection" in result.chunks[0].content
    assert result.chunks[0].verified_evidence == (
        "Broken pipe causes fuel leak at injector connection"
    )


def test_build_context_keeps_verified_quote_and_full_manual_passage():
    chunk = RetrievedChunk(
        content=(
            "Pump pressure does not increase. Check the relief valve. "
            "Check the hydraulic circuit for leaks. Check the pressure sensor."
        ),
        verified_evidence="Pump pressure does not increase",
        reference="13-25",
    )
    context = _build_context([chunk])
    assert "หลักฐานยืนยันความเกี่ยวข้อง: Pump pressure does not increase" in context
    assert "Check the hydraulic circuit for leaks" in context
    assert "Check the pressure sensor" in context


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


def test_parse_rerank_result_accepts_ordered_verbatim_table_cells():
    chunks = [
        RetrievedChunk(
            content=(
                "Slow boom up, insufficient power No.\n"
                "Pump pressure sensor\n"
                "Carry out service diagnosis for P1, P2 pump pressures in operation."
            ),
            reference="47-7",
        )
    ]
    raw = json.dumps(
        {
            "status": "answer",
            "selections": [
                {
                    "index": 1,
                    "confidence": 0.94,
                    "evidence": (
                        "Slow boom up, insufficient power No. | ... | "
                        "Pump pressure sensor | Carry out service diagnosis for P1, P2 "
                        "pump pressures in operation."
                    ),
                }
            ],
        }
    )
    result = _parse_rerank_result(raw, chunks)
    assert [chunk.reference for chunk in result.chunks] == ["47-7"]


def test_parse_rerank_result_accepts_reordered_verbatim_table_cells():
    chunks = [RetrievedChunk(content="First diagnostic cell\nSecond diagnostic cell")]
    raw = json.dumps(
        {
            "status": "answer",
            "selections": [
                {
                    "index": 1,
                    "confidence": 0.95,
                    "evidence": "Second diagnostic cell | First diagnostic cell",
                }
            ],
        }
    )
    assert len(_parse_rerank_result(raw, chunks).chunks) == 1


def test_parse_rerank_result_rejects_invented_table_cell():
    chunks = [RetrievedChunk(content="First diagnostic cell\nSecond diagnostic cell")]
    raw = json.dumps(
        {
            "status": "answer",
            "selections": [
                {
                    "index": 1,
                    "confidence": 0.95,
                    "evidence": "First diagnostic cell | Invented corrective action",
                }
            ],
        }
    )
    assert _parse_rerank_result(raw, chunks).chunks == []


def test_parse_rerank_result_accepts_high_confidence_page_with_semantic_support():
    chunks = [
        RetrievedChunk(
            content="P1 and P2 pump pressure checks. Check hydraulic pressure sensor.",
            reference="13-25",
        )
    ]
    raw = json.dumps(
        {
            "status": "answer",
            "selections": [
                {
                    "index": 1,
                    "confidence": 0.91,
                    "evidence": "Paraphrased evidence which is not verbatim",
                }
            ],
        }
    )
    result = _parse_rerank_result(
        raw,
        chunks,
        ["P1 pump pressure", "hydraulic pressure sensor"],
    )
    assert [chunk.reference for chunk in result.chunks] == ["13-25"]
    assert result.chunks[0].verified_evidence is None
    assert result.reason == "selected_semantic_support"


def test_safe_fallback_selects_best_supported_ranked_page():
    chunks = [
        RetrievedChunk(content="General maintenance information", reference="1-1"),
        RetrievedChunk(
            content="P1 pump pressure and hydraulic pressure sensor inspection",
            reference="13-25",
        ),
    ]
    fallback = _select_safe_fallback(
        chunks,
        ["P1 pump pressure", "hydraulic pressure sensor"],
    )
    assert fallback is not None
    assert fallback.reference == "13-25"
    assert fallback.verified_evidence is None


def test_safe_fallback_rejects_weak_single_term_overlap():
    # เคยลด threshold เหลือ 1 เพื่อเพิ่ม recall แต่พบจริงว่าดึงหน้าที่แค่มีคำซ้ำผิวเผินมาตอบผิดเรื่อง
    # (เช่น หน้าติดตั้งปั๊มใหม่มาตอบคำถามเรื่องปั๊มไม่สร้างแรงดัน) จึงกลับไปที่ 2 เหมือนเดิม
    chunk = RetrievedChunk(content="General pump maintenance", reference="1-1")
    assert _candidate_support_score(chunk, ["pump pressure sensor"]) == 1
    assert _select_safe_fallback([chunk], ["pump pressure sensor"]) is None


def test_safe_fallback_rejects_no_term_overlap():
    chunk = RetrievedChunk(content="Unrelated cabin air filter replacement", reference="1-1")
    assert _candidate_support_score(chunk, ["pump pressure sensor"]) == 0
    assert _select_safe_fallback([chunk], ["pump pressure sensor"]) is None


def test_safe_fallback_can_use_top_ranked_page_without_asking_back():
    chunk = RetrievedChunk(
        content="Technical manual passage for the closest retrieved topic",
        reference="10-1",
        score=0.04,
    )
    fallback = _select_safe_fallback(
        [chunk],
        [],
        allow_top_ranked=True,
    )
    assert fallback is not None
    assert fallback.reference == "10-1"


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
    assert result.clarification is None
    assert result.reason == "model_requested_clarification"


def test_clean_answer_removes_model_generated_source_labels():
    answer = (
        "ค่ามาตรฐาน 27.5 Ω [แหล่ง 17-43]\n"
        "ตรวจที่ขา 1-5 [source: page 17-43] (อ้างอิงจากคู่มือ)"
    )
    assert _clean_answer(answer) == "ค่ามาตรฐาน 27.5 Ω\nตรวจที่ขา 1-5"


def test_clean_answer_replaces_uncertain_cause_wording():
    answer = "สาเหตุอาจมาจากแรงดันปั๊มต่ำ ซึ่งอาจจะทำให้เครื่องช้า"
    assert _clean_answer(answer) == "สาเหตุเกิดจากแรงดันปั๊มต่ำ ซึ่งทำให้เครื่องช้า"


def test_clean_answer_strips_stray_headers_the_model_should_no_longer_use():
    answer = (
        "สาเหตุ: แรงดันปั๊มต่ำ อาจจะทำให้เครื่องช้า "
        "ตรวจสอบ: ตรวจแรงดันปั๊มและตรวจฟิลเตอร์ "
        "วิธีแก้: ปรับแรงดันปั๊มให้ได้ค่ามาตรฐาน"
    )
    cleaned = _clean_answer(answer)
    assert "สาเหตุ:" not in cleaned
    assert "ตรวจสอบ:" not in cleaned
    assert "วิธีแก้:" not in cleaned
    assert "อาจจะ" not in cleaned
    assert "แรงดันปั๊มต่ำ" in cleaned
    assert "ตรวจฟิลเตอร์" in cleaned
    assert "ปรับแรงดันปั๊มให้ได้ค่ามาตรฐาน" in cleaned


def test_is_insufficient_data_answer_detects_the_rule_8_boilerplate():
    answer = "สวัสดีครับช่างเต้ TIS ครับ\n\nไม่พบข้อมูลเพียงพอในฐานข้อมูล กรุณาระบุรุ่นเครื่องหรือ Error Code เพิ่มเติมครับ"
    assert is_insufficient_data_answer(answer)


def test_is_insufficient_data_answer_ignores_real_answers():
    answer = "เกิดจากแรงดันปั๊มต่ำ ให้ตรวจแรงดันปั๊มและตรวจฟิลเตอร์ครับ"
    assert not is_insufficient_data_answer(answer)


def test_ask_clarifying_question_returns_llm_generated_question():
    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMessage(content)

    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    async def fake_create(*args, **kwargs):
        return FakeResp("ตัวสั้นกับตัวยาวหมายถึงสายไฮดรอลิกเส้นไหนครับ")

    with patch(
        "app.services.claude_service._groq_client.chat.completions.create",
        side_effect=fake_create,
    ):
        result = asyncio.run(
            ask_clarifying_question("วิ่งซ้ายขวาไม่เท่ากัน ตัวสั้นกับตัวยาวเกี่ยวกันไหม")
        )
    assert result == "ตัวสั้นกับตัวยาวหมายถึงสายไฮดรอลิกเส้นไหนครับ"


def test_ask_clarifying_question_falls_back_when_llm_call_fails():
    async def fake_create(*args, **kwargs):
        raise RuntimeError("boom")

    with patch(
        "app.services.claude_service._groq_client.chat.completions.create",
        side_effect=fake_create,
    ):
        result = asyncio.run(ask_clarifying_question("คำถามอะไรสักอย่าง"))
    assert result == "ไม่พบข้อมูลเพียงพอในฐานข้อมูล กรุณาระบุชิ้นส่วนหรืออาการให้ชัดเจนขึ้นครับ"


def test_clean_answer_enforces_length_as_single_flowing_line():
    long_sentence = "ตรวจแรงดันปั๊มและตรวจฟิลเตอร์น้ำมันไฮดรอลิกทุกจุดตามลำดับในคู่มือ" * 12
    cleaned = _clean_answer(long_sentence)
    assert len(cleaned) <= 600
    assert "\n" not in cleaned
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


def test_search_category_hint_routes_common_question_shapes_without_hard_filter():
    assert infer_search_category_hint("ปั๊มมีแรงดันมาตรฐานเท่าไหร่") == "standard value measurement inspection"
    assert infer_search_category_hint("รถไม่มีแรงและมีเสียงหอน") == "troubleshooting cause diagnosis remedy"
    assert infer_search_category_hint("วิธีถอดปั๊มไฮดรอลิก") == "procedure inspection adjustment"
    assert infer_search_category_hint("รหัส P0217") == "error code diagnosis cause remedy"


def test_scope_filter_rejects_specific_actuator_for_general_machine_symptom():
    chunks = [
        RetrievedChunk(content="Section: troubleshooting\nSlow boom up, insufficient power"),
        RetrievedChunk(content="Section: pump adjustment\nP1 and P2 pump pressure checks"),
    ]
    filtered = _filter_scope_mismatches("รถทำงานช้าและแรงดันปั๊มต่ำ", chunks)
    assert [chunk.content for chunk in filtered] == [chunks[1].content]


def test_scope_filter_keeps_actuator_page_when_question_names_it():
    chunk = RetrievedChunk(content="Section: troubleshooting\nSlow boom up, insufficient power")
    assert _filter_scope_mismatches("บูมยกช้า", [chunk]) == [chunk]


def test_procedure_filter_rejects_installation_page_for_troubleshooting_question():
    chunks = [
        RetrievedChunk(content="Subsection: 33.4.8.3 INSTALLATION\nInstalling the pump..."),
        RetrievedChunk(content="Section: troubleshooting\nPump does not build up pressure..."),
    ]
    filtered = _filter_procedure_only_pages("ปั๊มไม่มีแรงเกิดจากอะไรครับ", chunks)
    assert [chunk.content for chunk in filtered] == [chunks[1].content]


def test_procedure_filter_keeps_installation_page_for_procedure_question():
    chunk = RetrievedChunk(content="Subsection: 33.4.8.3 INSTALLATION\nInstalling the pump...")
    assert _filter_procedure_only_pages("วิธีติดตั้งปั๊มใหม่", [chunk]) == [chunk]


def test_procedure_filter_does_not_empty_out_when_only_installation_pages_exist():
    chunk = RetrievedChunk(content="Subsection: 33.4.8.3 INSTALLATION\nInstalling the pump...")
    assert _filter_procedure_only_pages("ปั๊มไม่มีแรงเกิดจากอะไรครับ", [chunk]) == [chunk]


def test_build_page_image_urls_expands_every_page_the_chunk_covers():
    image_url = "https://x.supabase.co/storage/v1/object/public/manual-pages/sk2008_p1071.jpg"
    urls = _build_page_image_urls(image_url, [1071, 1072, 1073])
    assert urls == [
        "https://x.supabase.co/storage/v1/object/public/manual-pages/sk2008_p1071.jpg",
        "https://x.supabase.co/storage/v1/object/public/manual-pages/sk2008_p1072.jpg",
        "https://x.supabase.co/storage/v1/object/public/manual-pages/sk2008_p1073.jpg",
    ]


def test_build_page_image_urls_returns_none_for_single_page_chunk():
    image_url = "https://x.supabase.co/storage/v1/object/public/manual-pages/sk2008_p416.jpg"
    assert _build_page_image_urls(image_url, [416]) is None
    assert _build_page_image_urls(image_url, None) is None
    assert _build_page_image_urls(None, [416, 417]) is None


def test_build_page_image_urls_returns_none_when_filename_pattern_unrecognized():
    assert _build_page_image_urls("https://x.supabase.co/foo/bar.jpg", [1, 2]) is None
