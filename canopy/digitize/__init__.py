"""Figure digitizer: deterministic geometry for reading values off charts.

Task 6a (this package) is pure code — axis calibration, raster CV, exact vector geometry and overlays.
The vision-model routes that consume it (read-out, pixel coordinates, ensembles) live in `digitizer.py`.
"""
from .calibrate import (AxisCalibration, TickLabel, fit_axis, pair_ticks, parse_number, pixel_resolution,
                        px_to_value, value_to_px)
from .cv import (Axes, Bar, Marker, OcrResult, TickLabels, detect_bars, detect_markers, detect_whiskers,
                 find_axes, find_cap_ends, find_tick_marks, load_color, load_gray, ocr_tick_labels,
                 run_tesseract, snap_horizontal_edge, snap_vertical_edge, snap_window_for, tesseract_path)
from .overlay import draw_overlay
from .vector import (Mark, Segment, VectorScene, calibrate_from_scene, snap_to_vector, vector_candidates,
                     whisker_ends)
