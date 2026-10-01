"""Dashboard metrics computed from the live stream."""

import pytest

from python_ML_Metrics import MetricCollector


@pytest.mark.parametrize(
    "pins, saturated",
    [
        ([0, 0, 0, 0, 0, 0], 0),  # idle rig
        ([512, 300, 0, 700, 1, 1], 0),
        ([0, 0, 0, 0, 0, 1], 0),  # switch line is not a PWM channel
        ([1023, 0, 0, 0, 0, 0], 1),
    ],
)
def test_only_full_scale_counts_as_saturation(pins, saturated):
    collector = MetricCollector()
    collector.update_stream_from_backend(0.0, 1.0, pins)

    assert collector.dashboard_snapshot()["Saturation_Indicators_total"] == saturated


def test_an_idle_session_reports_no_saturation():
    collector = MetricCollector()

    for second in range(100):
        collector.update_stream_from_backend(float(second), 1.0, [0] * 6)

    assert collector.dashboard_snapshot()["Saturation_Indicators_total"] == 0
