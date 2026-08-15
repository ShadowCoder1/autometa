"""Task 6a: axis calibration (pixel -> data value) and tick pairing. Pure geometry, no LLM, offline."""
from __future__ import annotations

import math

import pytest

from canopy.digitize.calibrate import (
    AxisCalibration,
    TickLabel,
    fit_axis,
    pair_ticks,
    parse_number,
    pixel_resolution,
    px_to_value,
    value_to_px,
)


def _label(text: str, value: float | None, y: float, x: float = 40.0) -> TickLabel:
    return TickLabel(text=text, value=value, bbox=(x - 10, y - 5, x + 10, y + 5), center=(x, y))


# ------------------------------------------------------------------ fit_axis (linear)
def test_fit_axis_linear_recovers_a_and_b_exactly():
    a, b = -0.0725, 31.2                      # rows grow downward -> negative slope
    ticks = [(px, a * px + b) for px in (50.0, 120.0, 190.0, 260.0)]
    cal = fit_axis(ticks)
    assert cal.axis == "y" and cal.scale == "linear"
    assert cal.a == pytest.approx(a, rel=1e-9, abs=1e-12)
    assert cal.b == pytest.approx(b, rel=1e-9, abs=1e-12)
    assert cal.rmse < 1e-9
    assert cal.dropped == []
    assert len(cal.ticks) == 4


def test_px_to_value_and_value_to_px_round_trip():
    cal = fit_axis([(100.0, 20.0), (300.0, 0.0)])
    assert px_to_value(cal, 200.0) == pytest.approx(10.0)
    assert value_to_px(cal, 10.0) == pytest.approx(200.0)
    assert value_to_px(cal, px_to_value(cal, 137.5)) == pytest.approx(137.5)
    assert pixel_resolution(cal) == pytest.approx(0.1)   # data units per pixel


def test_fit_axis_x_axis_label():
    cal = fit_axis([(10.0, 1.0), (110.0, 2.0)], axis="x")
    assert cal.axis == "x"
    assert px_to_value(cal, 60.0) == pytest.approx(1.5)


# ------------------------------------------------------------------ fit_axis (log)
def test_fit_axis_log_recovers_decade_spacing():
    a, b = -0.01, 3.0                          # value = 10**(a*px + b)
    ticks = [(px, 10 ** (a * px + b)) for px in (0.0, 100.0, 200.0, 300.0)]
    cal = fit_axis(ticks, scale="log")
    assert cal.scale == "log"
    assert cal.a == pytest.approx(a, rel=1e-9)
    assert cal.b == pytest.approx(b, rel=1e-9)
    assert cal.rmse < 1e-9
    assert px_to_value(cal, 150.0) == pytest.approx(10 ** 1.5)
    assert value_to_px(cal, 100.0) == pytest.approx(100.0)
    # local resolution of a log axis depends on where you are
    assert pixel_resolution(cal, 100.0) == pytest.approx(abs(a) * math.log(10) * 100.0)
    assert pixel_resolution(cal, 200.0) < pixel_resolution(cal, 100.0)


def test_fit_axis_log_rejects_non_positive_values():
    with pytest.raises(ValueError):
        fit_axis([(0.0, 1.0), (100.0, 0.0)], scale="log")


def test_fit_axis_needs_two_ticks():
    with pytest.raises(ValueError):
        fit_axis([(10.0, 1.0)])


def test_fit_axis_rejects_duplicate_pixel():
    with pytest.raises(ValueError):
        fit_axis([(10.0, 1.0), (10.0, 2.0)])


# ------------------------------------------------------------------ robustness
def test_fit_axis_drops_one_bad_tick_and_reports_it():
    a, b = -0.05, 10.0
    ticks = [(px, a * px + b) for px in (20.0, 60.0, 100.0, 140.0, 180.0)]
    ticks[2] = (100.0, ticks[2][1] + 1.75)     # one OCR/tick blunder (35 px worth)
    cal = fit_axis(ticks)
    assert cal.dropped == [(100.0, pytest.approx(a * 100.0 + b + 1.75))]
    assert cal.a == pytest.approx(a, rel=1e-9)
    assert cal.b == pytest.approx(b, rel=1e-9)
    assert cal.rmse < 1e-9
    assert len(cal.ticks) == 4


def test_fit_axis_keeps_all_ticks_when_noise_is_comparable():
    a, b = -0.05, 10.0
    noise = [0.02, -0.03, 0.03, -0.02, 0.01]
    ticks = [(px, a * px + b + n) for px, n in zip((20.0, 60.0, 100.0, 140.0, 180.0), noise)]
    cal = fit_axis(ticks)
    assert cal.dropped == []
    assert len(cal.ticks) == 5
    assert 0 < cal.rmse < 1.0                  # rmse is reported in PIXELS
    assert cal.a == pytest.approx(a, abs=5e-4)


def test_fit_axis_never_drops_below_three_ticks():
    ticks = [(20.0, 1.0), (60.0, 2.0), (100.0, 9.0)]
    cal = fit_axis(ticks)                      # 3 ticks: no robust drop, keep them all
    assert cal.dropped == []
    assert len(cal.ticks) == 3


# ------------------------------------------------------------------ pair_ticks
def test_pair_ticks_uses_nearest_tick_line_position():
    labels = [_label("30", 30.0, 100.4), _label("20", 20.0, 150.6), _label("10", 10.0, 200.5), _label("0", 0.0, 250.4)]
    tick_lines = [100.0, 150.0, 200.0, 250.0]
    pairs = pair_ticks(labels, tick_lines)
    assert pairs == [(100.0, 30.0), (150.0, 20.0), (200.0, 10.0), (250.0, 0.0)]
    cal = fit_axis(pairs)
    assert cal.rmse < 1e-9


def test_pair_ticks_falls_back_to_label_centers():
    labels = [_label("30", 30.0, 100.0), _label("20", 20.0, 150.0), _label("10", 10.0, 200.0)]
    assert pair_ticks(labels, []) == [(100.0, 30.0), (150.0, 20.0), (200.0, 10.0)]


def test_pair_ticks_ignores_unparsed_labels_and_far_ticks():
    labels = [_label("30", 30.0, 100.0), _label("Time (s)", None, 150.0), _label("10", 10.0, 200.0)]
    tick_lines = [100.0, 200.0, 900.0]
    assert pair_ticks(labels, tick_lines) == [(100.0, 30.0), (200.0, 10.0)]


def test_pair_ticks_rejects_a_misread_label():
    # "10" mis-OCR'd as "70": its position is inconsistent with the other four
    labels = [_label("40", 40.0, 100.0), _label("30", 30.0, 150.0), _label("20", 20.0, 200.0),
              _label("70", 70.0, 250.0), _label("0", 0.0, 300.0)]
    tick_lines = [100.0, 150.0, 200.0, 250.0, 300.0]
    pairs = pair_ticks(labels, tick_lines)
    assert (250.0, 70.0) not in pairs
    assert pairs == [(100.0, 40.0), (150.0, 30.0), (200.0, 20.0), (300.0, 0.0)]


def test_pair_ticks_on_x_axis_uses_x_centers():
    labels = [TickLabel("1", 1.0, (10, 90, 30, 100), (20.0, 95.0)),
              TickLabel("2", 2.0, (70, 90, 90, 100), (80.0, 95.0))]
    assert pair_ticks(labels, [21.0, 79.0], axis="x") == [(21.0, 1.0), (79.0, 2.0)]


def test_pair_ticks_needs_at_least_two_pairs():
    assert pair_ticks([_label("10", 10.0, 100.0)], [100.0]) == []


# ------------------------------------------------------------------ numeric parsing
@pytest.mark.parametrize(
    "text,expected",
    [
        ("0", 0.0),
        ("-10", -10.0),
        ("−10", -10.0),          # unicode minus
        ("–10", -10.0),          # en dash used as minus
        ("1,234", 1234.0),
        ("1 234", 1234.0),       # thin space as thousands separator
        ("12.5", 12.5),
        ("1,5", 1.5),                 # decimal comma
        ("2.5%", 2.5),
        ("1e3", 1000.0),
        ("  40 ", 40.0),
        ("+7", 7.0),
        ("Time (s)", None),
        ("", None),
        ("--", None),
        ("1.2.3", None),
    ],
)
def test_parse_number(text, expected):
    got = parse_number(text)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_axis_calibration_is_serializable():
    cal = fit_axis([(100.0, 20.0), (300.0, 0.0)])
    assert isinstance(cal, AxisCalibration)
    d = cal.to_dict()
    assert d["axis"] == "y" and d["scale"] == "linear"
    assert d["ticks"] == [[100.0, 20.0], [300.0, 0.0]]
