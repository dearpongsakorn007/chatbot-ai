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


async def retrieve_chunks(query_embedding: list[float]) -> list[RetrievedChunk]:
    supabase = get_supabase()
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": settings.top_k,
            "filter": {},
        },
    ).execute()

    rows = result.data or []
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
                score=row.get("similarity"),
                reference=str(reference) if reference is not None else None,
                image_url=metadata.get("image_url") or image.get("image_url"),
                preview_image_url=(
                    metadata.get("preview_image_url")
                    or image.get("preview_image_url")
                ),
            )
        )
    return chunks
