"""Ingestion CLI: parse -> chunk -> embed -> persist.

Run once before starting the server:

    python -m docqa.ingest            # build index (skips if nothing changed)
    python -m docqa.ingest --force    # rebuild from scratch

Outputs (under CHROMA_DIR):
  - Chroma collection with embeddings + metadata (vector search)
  - chunks.json   : all chunk records, used to rebuild the BM25 index at query time
  - manifest.json : per-file content hashes for idempotency
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

from .chunking import Chunk, chunk_document
from .config import Settings, get_settings
from .doc_store import build_doc_records, save_doc_records
from .embeddings import build_embedder
from .pdf_parser import ParsedDocument, parse_pdf
from .vector_store import VectorStore


def _file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _discover_pdfs(settings: Settings) -> list:
    return sorted(settings.data_path.glob("*.pdf"))


def _current_manifest(pdfs: list, settings: Settings) -> dict:
    return {
        "embed_provider": settings.embed_provider,
        "embed_model": settings.embed_model
        if settings.embed_provider == "openai"
        else settings.local_embed_model,
        "chunk_target_tokens": settings.chunk_target_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "files": {p.name: _file_hash(p) for p in pdfs},
    }


def _unchanged(settings: Settings, manifest: dict) -> bool:
    if not settings.manifest_path.exists():
        return False
    try:
        prev = json.loads(settings.manifest_path.read_text())
    except Exception:
        return False
    return prev == manifest and settings.chunks_path.exists() and settings.documents_path.exists()


def build_index(settings: Settings, force: bool = False) -> int:
    pdfs = _discover_pdfs(settings)
    if not pdfs:
        raise SystemExit(f"No PDFs found in {settings.data_path}")

    manifest = _current_manifest(pdfs, settings)
    if not force and _unchanged(settings, manifest):
        store = VectorStore(settings)
        print(f"Index already up to date ({store.count()} chunks). Use --force to rebuild.")
        return store.count()

    print(f"Parsing + chunking {len(pdfs)} document(s)...")
    all_chunks: list[Chunk] = []
    parsed_docs: list[ParsedDocument] = []
    for pdf in pdfs:
        parsed = parse_pdf(pdf)
        parsed_docs.append(parsed)
        chunks = chunk_document(
            parsed, settings.chunk_target_tokens, settings.chunk_overlap_tokens
        )
        all_chunks.extend(chunks)
        print(f"  {parsed.meta.company:32} {parsed.page_count:3d}p -> {len(chunks):4d} chunks")

    print(f"Embedding {len(all_chunks)} chunks via {settings.embed_provider}...")
    t0 = time.time()
    embedder = build_embedder(settings)
    embeddings = embedder.embed_documents([c.text for c in all_chunks])
    print(f"  embedded in {time.time() - t0:.1f}s (dim={len(embeddings[0])})")

    print("Writing vector store + chunk records...")
    store = VectorStore(settings)
    store.reset_collection()
    store.add(
        ids=[c.id for c in all_chunks],
        embeddings=embeddings,
        documents=[c.text for c in all_chunks],
        metadatas=[c.metadata() for c in all_chunks],
    )

    records = [{"id": c.id, "text": c.text, **c.metadata()} for c in all_chunks]
    settings.chunks_path.write_text(json.dumps(records))
    save_doc_records(settings, build_doc_records(parsed_docs))
    settings.manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Done. Indexed {len(all_chunks)} chunks from {len(pdfs)} documents.")
    return len(all_chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the DocQA index.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if unchanged.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    build_index(get_settings(), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
