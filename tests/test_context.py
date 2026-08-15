"""Task 3: context builders (page/figure blocks) + Files API upload. Offline only."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from canopy.ingest.pdf import PaperRecord, ingest_pdf
from canopy.llm.cache import cache_key
from canopy.llm.client import LLMClient
from canopy.llm.context import excerpt_for, figure_blocks, page_blocks, upload_pdf
from canopy.llm.providers import FakeProvider

PDF = Path(__file__).resolve().parent / "fixtures" / "pdfs" / "bock2005.pdf"


@pytest.fixture(scope="session")
def paper(tmp_path_factory) -> PaperRecord:
    out = tmp_path_factory.mktemp("bock2005")
    return ingest_pdf(PDF, out)


def _png_bytes(paper: PaperRecord, rel: str) -> bytes:
    return (Path(paper.out_dir) / rel).read_bytes()


def _decoded(block: dict) -> bytes:
    return base64.standard_b64decode(block["source"]["data"])


# ------------------------------------------------------------------ ingestion sanity
def test_fixture_pdf_ingests(paper):
    assert paper.n_pages == 5
    assert paper.has_text_layer
    assert [f.id for f in paper.figures][:2] == ["fig01", "fig02"]


# ------------------------------------------------------------------ page blocks
def test_page_blocks_two_per_page_in_order(paper):
    blocks = page_blocks(paper, [1, 2])
    assert len(blocks) == 4
    assert [b["type"] for b in blocks] == ["text", "image", "text", "image"]
    assert blocks[0]["text"].startswith("[page 1 text]")
    assert blocks[2]["text"].startswith("[page 2 text]")
    assert "Exp Brain Res" in blocks[0]["text"]
    assert paper.page_text(2)[:60] in blocks[2]["text"]
    assert _decoded(blocks[1]) == _png_bytes(paper, "pages/p001.png")
    assert _decoded(blocks[3]) == _png_bytes(paper, "pages/p002.png")
    assert _decoded(blocks[1]).startswith(b"\x89PNG")
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert all("cache_control" not in b for b in blocks)


def test_page_blocks_text_or_image_only(paper):
    only_images = page_blocks(paper, [3], with_text=False)
    assert len(only_images) == 1 and only_images[0]["type"] == "image"
    only_text = page_blocks(paper, [3], with_images=False)
    assert len(only_text) == 1 and only_text[0]["type"] == "text"


def test_page_blocks_validate_page_numbers(paper):
    with pytest.raises(ValueError):
        page_blocks(paper, [0])
    with pytest.raises(ValueError):
        page_blocks(paper, [99])


# ------------------------------------------------------------------ figure blocks
def test_figure_blocks_caption_then_image(paper):
    blocks = figure_blocks(paper, ["fig01"])
    assert [b["type"] for b in blocks] == ["text", "image"]
    assert "fig01" in blocks[0]["text"]
    assert "page 3" in blocks[0]["text"]
    assert "Fig. 1" in blocks[0]["text"]
    assert _decoded(blocks[1]) == _png_bytes(paper, "figures/fig01.claude.png")


def test_figure_blocks_unknown_id(paper):
    with pytest.raises(KeyError):
        figure_blocks(paper, ["fig99"])


# ------------------------------------------------------------------ excerpts
def test_excerpt_for_caches_only_the_last_block(paper):
    blocks = excerpt_for(paper, pages=[3], figures=["fig01"])
    assert len(blocks) == 4
    assert [b["type"] for b in blocks] == ["text", "image", "text", "image"]
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in b for b in blocks[:-1])


def test_excerpt_for_without_cache(paper):
    blocks = excerpt_for(paper, pages=[3], figures=[], cache=False)
    assert blocks and all("cache_control" not in b for b in blocks)


def test_excerpt_blocks_are_json_serialisable_and_hashable(paper):
    blocks = excerpt_for(paper, pages=[3], figures=["fig01"])
    json.dumps(blocks)
    msgs = [{"role": "user", "content": blocks}]
    k1 = cache_key(model="claude-opus-5", system="s", messages=msgs)
    k2 = cache_key(model="claude-opus-5", system="s",
                   messages=[{"role": "user", "content": excerpt_for(paper, pages=[3],
                                                                     figures=["fig01"])}])
    assert k1 == k2                                    # stable
    other = [{"role": "user", "content": excerpt_for(paper, pages=[4], figures=["fig01"])}]
    assert cache_key(model="claude-opus-5", system="s", messages=other) != k1


def test_excerpt_for_empty_selection(paper):
    assert excerpt_for(paper, pages=[], figures=[]) == []


# ------------------------------------------------------------------ Files API
def test_upload_pdf_uses_cached_file_id(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7\n%%EOF\n")
    (tmp_path / "file_id.txt").write_text("file_cached_123\n")
    provider = FakeProvider()
    client = LLMClient(cache_dir=None, provider=provider)
    assert upload_pdf(client, pdf) == "file_cached_123"
    assert provider.uploads == []                      # no API traffic


def test_upload_pdf_uploads_once_and_persists(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\n%%EOF\n")
    out = tmp_path / "ingest"
    out.mkdir()
    provider = FakeProvider()
    client = LLMClient(cache_dir=None, provider=provider)
    file_id = upload_pdf(client, pdf, out_dir=out)
    assert file_id.startswith("file_")
    assert (out / "file_id.txt").read_text().strip() == file_id
    assert len(provider.uploads) == 1
    assert upload_pdf(client, pdf, out_dir=out) == file_id
    assert len(provider.uploads) == 1                  # cached, not re-uploaded


def test_upload_pdf_for_ingested_paper_defaults_to_out_dir(paper):
    provider = FakeProvider()
    client = LLMClient(cache_dir=None, provider=provider)
    file_id = upload_pdf(client, PDF, out_dir=paper.out_dir)
    assert (Path(paper.out_dir) / "file_id.txt").read_text().strip() == file_id
    assert provider.uploads[0]["betas"] == ["files-api-2025-04-14"]
