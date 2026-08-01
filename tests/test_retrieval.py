from app.services.retrieval_service import _expand_search_query


def test_expand_thai_arm_load_terms():
    query = _expand_search_query("SK200-8 อาร์มเข้า อาร์มออกมีอาการโหลด")
    assert query == "arm in out load sk200-8"


def test_expand_common_machine_terms():
    query = _expand_search_query("บูมช้า ปั๊มไม่มีแรง")
    assert query == "slow weak boom pump"


def test_unmapped_question_skips_fulltext_search():
    assert _expand_search_query("ขอรายละเอียดเพิ่มเติม") == ""
