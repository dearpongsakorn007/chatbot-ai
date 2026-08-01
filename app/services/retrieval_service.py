"""
ค้นหา chunks ที่เกี่ยวข้องกับคำถามผู้ใช้จาก Supabase pgvector
ใช้ function/RPC ชื่อ match_documents ซึ่งรองรับ embeddings ชุดปัจจุบัน
"""
import logging

from app.db.supabase_client import get_supabase
from app.config import settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger("repair-bot")

def _find_reference_images(supabase, rows: list[dict]) -> dict[tuple[str, str], dict]:
    pages_by_source: dict[str, set[int | str]] = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        source_file = metadata.get("source_file")
        pdf_page = metadata.get("pdf_page_start")
        if source_file and pdf_page is not None:
            pages_by_source.setdefault(source_file, set()).add(pdf_page)

    images_by_page: dict[tuple[str, str], dict] = {}
    for source_file, pdf_pages in pages_by_source.items():
        try:
            image_result = (
                supabase.table("documents_gemini")
                .select("metadata")
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
            key = (source_file, str(pdf_page))
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
) -> list[RetrievedChunk]:
    supabase = get_supabase()
    candidate_count = max(settings.top_k * 2, 10)
    vector_result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": candidate_count,
            "filter": {},
        },
    ).execute()

    result_groups: list[list[dict]] = []
    for fulltext_query in (search_queries or [])[:3]:
        try:
            fulltext_result = supabase.rpc(
                "search_sk2008_fulltext",
                {
                    "search_query": fulltext_query,
                    "result_limit": candidate_count,
                },
            ).execute()
            result_groups.append(fulltext_result.data or [])
        except Exception as exc:
            logger.warning("full-text search failed: %s", exc)

    # Interleave exact, alternative, and semantic results so one broad query cannot
    # consume every context slot before the other retrieval strategies are considered.
    result_groups.append(vector_result.data or [])
    rows: list[dict] = []
    for rank in range(candidate_count):
        for group in result_groups:
            if rank < len(group):
                rows.append(group[rank])

    unique_rows = []
    seen_ids = set()
    for row in rows:
        row_id = row.get("id")
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        unique_rows.append(row)
        if len(unique_rows) >= settings.top_k:
            break
    rows = unique_rows
    reference_images = _find_reference_images(supabase, rows)
    chunks = []
    for row in rows:
        metadata = row.get("metadata") or {}
        reference = (
            metadata.get("page_reference")
            or metadata.get("manual_page")
            or metadata.get("pdf_page_start")
        )
        image = reference_images.get(
            (str(metadata.get("source_file") or ""), str(metadata.get("pdf_page_start")))
        ) or {}
        chunks.append(
            RetrievedChunk(
                content=row.get("content", ""),
                source=(
                    row.get("source")
                    or metadata.get("source_file")
                    or metadata.get("source")
                ),
                score=row.get("similarity") or row.get("search_score"),
                reference=str(reference) if reference is not None else None,
                image_url=metadata.get("image_url") or image.get("image_url"),
                preview_image_url=(
                    metadata.get("preview_image_url")
                    or image.get("preview_image_url")
                ),
            )
        )
    return chunks
