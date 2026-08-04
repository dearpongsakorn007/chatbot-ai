"""
ค้นหา chunks ที่เกี่ยวข้องกับคำถามผู้ใช้จาก Supabase pgvector
ใช้ function/RPC ชื่อ match_documents ซึ่งรองรับ embeddings ชุดปัจจุบัน
"""
import logging

from app.db.supabase_client import get_supabase
from app.config import settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger("repair-bot")
RRF_K = 60
FULLTEXT_WEIGHTS = (3.0, 1.5, 1.0)
VECTOR_WEIGHT = 2.0
# ต้องตรงกับ document_id_exact ที่ match_documents/search_sk2008_fulltext ใช้กรองอยู่แล้ว
# กันไม่ให้รูปอ้างอิงหลุดไปหยิบจากข้อมูลชุดเก่า (v1/v2/v3/...) ที่เลขหน้าไม่ตรงกับ v4
CURRENT_DOCUMENT_ID = "kobelco-sk2008-repair-ocr-v4"
# บอทตัวนี้รองรับคู่มือรุ่นเดียว ตรงกับ CURRENT_DOCUMENT_ID ด้านบน
CURRENT_MACHINE_MODEL = "SK200-8"


def _build_error_code_content(row: dict) -> str:
    """ประกอบเนื้อหาจากคอลัมน์ที่ตรวจสอบแล้ว (ไม่ใช้ source_content ซึ่งเป็น OCR ดิบที่ยังเพี้ยนอยู่)"""
    lines = [f"Error code: {row.get('code')} ({row.get('brand', '')} {row.get('model', '')})".strip()]
    if row.get("meaning"):
        lines.append(f"Meaning: {row['meaning']}")
    if row.get("judgment_condition"):
        lines.append(f"Judging condition: {row['judgment_condition']}")
    if row.get("symptom"):
        lines.append(f"Symptom: {row['symptom']}")
    if row.get("failure_control"):
        lines.append(f"Control in the event of failure: {row['failure_control']}")
    if row.get("normal_condition"):
        lines.append(f"Normal condition after correction: {row['normal_condition']}")
    related_parts = row.get("related_parts") or []
    if related_parts:
        lines.append("Related parts: " + ", ".join(related_parts))
    check_steps = row.get("check_steps") or []
    if check_steps:
        lines.append("Checking steps:")
        for index, step in enumerate(check_steps, start=1):
            lines.append(f"{index}. {step}")
    if row.get("diagnosis_screen"):
        lines.append(f"Service diagnosis screen: {row['diagnosis_screen']}")
    return "\n".join(lines)


async def lookup_error_code(code: str) -> RetrievedChunk | None:
    """ค้นหา error code ตรง ๆ จากตารางที่ตรวจสอบแล้ว (error_codes_sk2008)

    ใช้ก่อนเสมอเมื่อคำถามระบุ error code ชัดเจน เพราะข้อมูลผ่าน QA/verified แล้ว
    แม่นยำกว่า chunk ที่ OCR มาจาก PDF ตรง ๆ ใน documents_gemini มาก
    คืนค่า None เมื่อไม่พบ ให้ retrieval แบบเดิม (vector/full-text) ทำงานต่อแทน
    """
    supabase = get_supabase()
    try:
        result = (
            supabase.table("error_codes_sk2008")
            .select("*")
            .eq("code", code.upper())
            .eq("model", CURRENT_MACHINE_MODEL)
            .eq("verified", True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.warning("error_codes_sk2008 lookup failed: %s", exc)
        return None

    rows = result.data or []
    if not rows:
        return None
    row = rows[0]
    return RetrievedChunk(
        content=_build_error_code_content(row),
        verified_evidence=f"Verified error code table entry for {row.get('code')}",
        source=f"{row.get('source_file') or 'error_codes_sk2008'} (verified error code table)",
        score=1.0,
        reference=row.get("manual_page"),
        image_url=row.get("image_url"),
        preview_image_url=row.get("image_url"),
    )


def _row_key(row: dict) -> str:
    row_id = row.get("id")
    if row_id is not None:
        return f"id:{row_id}"
    metadata = row.get("metadata") or {}
    return "|".join(
        (
            str(metadata.get("source_file") or row.get("source") or ""),
            str(metadata.get("page_reference") or metadata.get("pdf_page_start") or ""),
            str(row.get("content") or ""),
        )
    )


def _result_score(row: dict) -> float:
    value = row.get("search_score")
    if value is None:
        value = row.get("similarity")
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _stable_group(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (-_result_score(row), _row_key(row)))


def _rank_results(
    result_groups: list[tuple[float, list[dict]]],
    limit: int,
) -> list[dict]:
    """Deterministic weighted reciprocal-rank fusion for hybrid retrieval."""
    fused_scores: dict[str, float] = {}
    rows_by_key: dict[str, dict] = {}

    for weight, rows in result_groups:
        for rank, row in enumerate(_stable_group(rows), start=1):
            key = _row_key(row)
            rows_by_key.setdefault(key, row)
            fused_scores[key] = fused_scores.get(key, 0.0) + weight / (RRF_K + rank)

    ranked_keys = sorted(fused_scores, key=lambda key: (-fused_scores[key], key))
    ranked_rows = []
    for key in ranked_keys[:limit]:
        row = dict(rows_by_key[key])
        row["fusion_score"] = fused_scores[key]
        ranked_rows.append(row)
    return ranked_rows


def _find_reference_images(supabase, rows: list[dict]) -> dict[tuple[str, str], dict]:
    pages_by_source: dict[str, set[int | str]] = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        source_file = metadata.get("source_file")
        pdf_page = metadata.get("pdf_page_start")
        if source_file and pdf_page is not None:
            pages_by_source.setdefault(source_file, set()).add(pdf_page)

    images_by_page: dict[tuple[str, str, str], dict] = {}
    for source_file, pdf_pages in pages_by_source.items():
        try:
            image_result = (
                supabase.table("documents_gemini")
                .select("metadata")
                # ล็อกเวอร์ชันข้อมูลให้ตรงกับที่ retrieve_chunks ใช้ค้นหาจริง
                # ป้องกันไม่ให้หยิบรูปจากข้อมูลชุดเก่าที่เลขหน้าไม่ได้อ้างอิงเนื้อหาเดียวกัน
                .eq("metadata->>document_id", CURRENT_DOCUMENT_ID)
                .eq("metadata->>source_file", source_file)
                .in_("metadata->>pdf_page_start", sorted(pdf_pages, key=str))
                .not_.is_("metadata->>image_url", "null")
                .limit(50)
                .execute()
            )
        except Exception as exc:
            logger.warning("reference image lookup failed: %s", exc)
            continue

        for image_row in image_result.data or []:
            metadata = image_row.get("metadata") or {}
            image_url = metadata.get("image_url")
            pdf_page = metadata.get("pdf_page_start")
            if not image_url or pdf_page is None:
                continue
            # ถ้าแถวนั้นมีรหัส error code ให้ผูกรูปเข้ากับรหัสนั้นโดยเฉพาะ
            # กันกรณีหน้าเดียวกันมีหลาย error code/หัวข้อปนกัน (matched by page only ผิดได้)
            code = str(metadata.get("code") or "")
            key = (source_file, str(pdf_page), code)
            images_by_page.setdefault(
                key,
                {
                    "image_url": image_url,
                    "preview_image_url": metadata.get("preview_image_url") or image_url,
                },
            )

    return images_by_page


async def retrieve_chunks(
    query_embedding: list[float],
    search_queries: list[str] | None = None,
    content_type_filter: list[str] | None = None,
    section_code_filter: list[str] | None = None,
) -> list[RetrievedChunk]:
    supabase = get_supabase()
    # ดึงกว้างกว่าจำนวนที่จะส่งให้ reranker เพื่อไม่ให้หน้าที่ถูกต้องหลุดจาก top 5 เร็วเกินไป
    candidate_count = max(settings.top_k * 3, 30)
    vector_result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": candidate_count,
            "filter": {},
            "content_type_filter": content_type_filter,
            "section_code_filter": section_code_filter,
        },
    ).execute()

    result_groups: list[tuple[float, list[dict]]] = []
    fulltext_counts: list[int] = []
    for query_index, fulltext_query in enumerate((search_queries or [])[:3]):
        try:
            fulltext_result = supabase.rpc(
                "search_sk2008_fulltext",
                {
                    "search_query": fulltext_query,
                    "result_limit": candidate_count,
                    "content_type_filter": content_type_filter,
                    "section_code_filter": section_code_filter,
                },
            ).execute()
            result_groups.append(
                (FULLTEXT_WEIGHTS[query_index], fulltext_result.data or [])
            )
            fulltext_counts.append(len(fulltext_result.data or []))
        except Exception as exc:
            logger.warning("full-text search failed: %s", exc)

    result_groups.append((VECTOR_WEIGHT, vector_result.data or []))
    rows = _rank_results(result_groups, settings.top_k)
    logger.info(
        "retrieval completed vector=%d fulltext=%s ranked=%d content_types=%s sections=%s",
        len(vector_result.data or []),
        fulltext_counts,
        len(rows),
        content_type_filter,
        section_code_filter,
    )
    reference_images = _find_reference_images(supabase, rows)
    chunks = []
    for row in rows:
        metadata = row.get("metadata") or {}
        reference = (
            metadata.get("page_reference")
            or metadata.get("manual_page")
            or metadata.get("pdf_page_start")
        )
        source_file = str(metadata.get("source_file") or "")
        pdf_page = str(metadata.get("pdf_page_start"))
        code = str(metadata.get("code") or "")
        # หา image ที่ผูกกับ error code เดียวกันก่อน ถ้าไม่มีค่อย fallback เป็นรูปของหน้านั้นแบบไม่ระบุ code
        image = (
            reference_images.get((source_file, pdf_page, code))
            or reference_images.get((source_file, pdf_page, ""))
            or {}
        )
        chunks.append(
            RetrievedChunk(
                content=row.get("content", ""),
                source=(
                    row.get("source")
                    or metadata.get("source_file")
                    or metadata.get("source")
                ),
                score=(
                    row.get("fusion_score")
                    or row.get("similarity")
                    or row.get("search_score")
                ),
                reference=str(reference) if reference is not None else None,
                image_url=metadata.get("image_url") or image.get("image_url"),
                preview_image_url=(
                    metadata.get("preview_image_url")
                    or image.get("preview_image_url")
                ),
            )
        )
    return chunks
