"""Deterministic PDF ingestion (dedupe, parse, raster, figure regions)."""
from .pdf import ingest_pdf, PaperRecord, FigureRegion, PanelRegion, TableRecord, PageRecord
from .dedupe import dedupe_pdfs, PaperGroup
from .images import prepare_for_claude, resized_size, count_image_tokens
