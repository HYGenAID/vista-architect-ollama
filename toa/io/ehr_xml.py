
from __future__ import annotations
from typing import Iterator
from .xml_filter import get_streamlined_chunk


def iter_chunks(xml_text: str, max_chars: int = 120_000, min_final_chunk_ratio: float = 0.8, strategy: str = "streamlined") -> Iterator[str]:
    """
    Char-based chunker with optional streamlined filtering.

    Two strategies available:

    **"streamlined"** (default):
    1. Reserve final chunk (approximately max_chars, at least min_final_chunk_ratio * max_chars)
    2. Filter beginning portion to notes/procedures/conditions only
    3. Chunk filtered beginning into max_chars chunks
    4. Yield filtered early chunks, then full final chunk

    **"full"** (legacy):
    Simple sequential chunking - all chunks contain complete XML without filtering

    Args:
        xml_text: Complete EHR XML text
        max_chars: Target chunk size (default 120k chars ≈ 30k tokens)
        min_final_chunk_ratio: Minimum size ratio for final chunk (default 0.8, only used in streamlined mode)
        strategy: "streamlined" or "full" (default "streamlined")

    Yields:
        XML chunks: filtered early chunks + full final chunk (streamlined), or sequential full chunks (full)
    """
    # Legacy "full" strategy: simple sequential chunking
    if strategy == "full":
        for i in range(0, len(xml_text), max_chars):
            yield xml_text[i:i+max_chars]
        return

    # Streamlined strategy: filter early chunks, preserve final chunk
    total_length = len(xml_text)
    min_final_size = int(max_chars * min_final_chunk_ratio)

    # If entire XML fits in one chunk, yield as-is (no filtering needed)
    if total_length <= max_chars:
        yield xml_text
        return

    # Determine final chunk boundary (take ~max_chars from end)
    final_chunk_start = total_length - max_chars

    # Ensure final chunk meets minimum size requirement
    if final_chunk_start < 0 or (total_length - final_chunk_start) < min_final_size:
        final_chunk_start = max(0, total_length - min_final_size)

    # Split: beginning (to be filtered) + final (full XML)
    beginning = xml_text[:final_chunk_start]
    final_chunk = xml_text[final_chunk_start:]

    # Filter beginning to notes/procedures/conditions only
    beginning_filtered = get_streamlined_chunk(beginning)

    # Chunk the filtered beginning into max_chars chunks
    for i in range(0, len(beginning_filtered), max_chars):
        chunk = beginning_filtered[i:i+max_chars]
        if chunk.strip():  # Only yield non-empty chunks
            yield chunk

    # Finally, yield the complete final chunk (unfiltered)
    yield final_chunk
