"""Everything a reader sees: forest plots, tables, PRISMA flow, provenance, the HTML report.

Nothing in this package computes a statistic. Every number it prints was produced by
`canopy.stats` (pooling, heterogeneity, Egger, leave-one-out) or transcribed by an agent and
verified in `canopy.verify`; the report's whole job is to show those numbers together with the
conventions and the provenance that make them checkable.
"""
from .forest import forest_layout, forest_plot
from .html import human_review_table, methods_paragraph, write_html_report
from .methods_fig import methods_figure, route_counts, route_examples, route_group
from .outputs import outcome_dir, write_outcome_outputs
from .provenance import figure_provenance, provenance_bundle, quote_crop
from .tables import (EXCLUSION_REASONS, EXTRACTION_COLUMNS, SENSITIVITY_ANALYSES,
                     exclusions_table, extraction_table, funnel_plot, leave_one_out_table,
                     pool_rows, prisma_flow, sensitivity_analyses, sensitivity_outputs,
                     write_rows)

__all__ = ["forest_plot", "forest_layout", "extraction_table", "EXTRACTION_COLUMNS",
           "exclusions_table", "EXCLUSION_REASONS", "leave_one_out_table", "sensitivity_analyses",
           "sensitivity_outputs", "SENSITIVITY_ANALYSES", "funnel_plot", "prisma_flow",
           "pool_rows", "write_rows", "methods_figure", "route_counts", "route_examples",
           "route_group", "quote_crop", "figure_provenance", "provenance_bundle",
           "write_html_report", "methods_paragraph", "human_review_table",
           "write_outcome_outputs", "outcome_dir"]
