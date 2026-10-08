import math

import pandas as pd
import pytest

from python_Autonomy_Guard import AutonomyGuard, ControlEnvelope


@pytest.mark.req("AUTO-01")
@pytest.mark.parametrize(
    "validation, admitted, reason",
    [
        ({"validation_r2": 0.77}, True, ""),
        ({"validation_r2": 0.5}, True, ""),
        ({"validation_r2": 0.49}, False, "below the 0.50 floor"),
        ({"validation_r2": -6.0}, False, "below the 0.50 floor"),
        ({"validation_r2": float("nan")}, False, "below the 0.50 floor"),
        ({"validation_r2": "high"}, False, "not a number"),
        ({}, False, "no held-out validation score"),
        (None, False, "no held-out validation score"),
    ],
)
def test_only_a_validated_model_may_drive_the_rig(validation, admitted, reason):
    ok, why = AutonomyGuard().model_admissible(validation)
    assert ok is admitted and reason in why


@pytest.mark.req("AUTO-02")
def test_a_proposal_inside_the_limits_is_sent_as_it_is():
    guard = AutonomyGuard()
    review = guard.review([1.1, 1.0, 0.9, 1.2, 0.8], [1.0] * 5)

    assert review.action == "accept" and review.targets == pytest.approx([1.1, 1.0, 0.9, 1.2, 0.8])
    assert guard.counts.accepted == 1


@pytest.mark.req("AUTO-02")
def test_each_channel_moves_at_most_one_step_and_stays_in_the_envelope():
    guard = AutonomyGuard(ControlEnvelope(max_step_v=0.25))
    review = guard.review([3.3, 0.0, 9.0, -9.0, 1.0], [1.0, 1.0, 3.2, 0.1, 1.0])

    assert review.action == "limit"
    assert review.targets == pytest.approx([1.25, 0.75, 3.3, 0.0, 1.0])
    assert "pin 3" in review.reason or "pin 4" in review.reason
    assert guard.counts.limited == 1


@pytest.mark.req("AUTO-02")
def test_a_channel_already_outside_the_envelope_is_brought_back():
    review = AutonomyGuard(ControlEnvelope(minimum_v=(0.5,) * 5, maximum_v=(2.0,) * 5)).review([2.5] * 5, [2.4] * 5)
    assert review.targets == pytest.approx([2.0] * 5)


@pytest.mark.req("AUTO-02")
@pytest.mark.parametrize("proposal", [[1.0, math.nan, 1.0, 1.0, 1.0], [1.0, math.inf, 1.0, 1.0, 1.0], [1.0] * 4, ["a"] * 5, None])
def test_a_malformed_proposal_is_rejected_and_nothing_is_sent(proposal):
    guard = AutonomyGuard()
    review = guard.review(proposal, [1.0] * 5)

    assert review.action == "reject" and review.targets is None
    assert guard.counts.rejected == 1


def frame(pins, voltage, rows=60):
    return pd.DataFrame({**{f"pin_{i + 1}": [pins[i]] * rows for i in range(5)}, "voltage": [voltage] * rows})


STATS = {"columns": ["pin_1", "pin_2", "pin_3", "pin_4", "pin_5", "voltage"],
         "mean": [500.0, 500.0, 500.0, 500.0, 500.0, 1.5], "scale": [100.0] * 5 + [0.2]}


@pytest.mark.req("AUTO-03")
def test_inputs_inside_the_training_data_are_not_drift():
    guard = AutonomyGuard(drift_z_limit=4.0)
    assert guard.drift(frame([520, 480, 600, 450, 500], 1.6), STATS) == (False, "")


@pytest.mark.req("AUTO-03")
def test_inputs_far_outside_the_training_data_are_drift_and_named():
    guard = AutonomyGuard(drift_z_limit=4.0)
    drifted, reason = guard.drift(frame([500, 500, 1000, 500, 500], 1.5), STATS)

    assert drifted and reason.startswith("pin_3 is 5.0 training standard deviations")
    assert guard.counts.held_for_drift == 1


@pytest.mark.req("AUTO-03")
def test_drift_looks_only_at_the_recent_window():
    old = frame([1000] * 5, 3.0, rows=200)
    recent = frame([500] * 5, 1.5, rows=40)
    assert AutonomyGuard(drift_window=40).drift(pd.concat([old, recent], ignore_index=True), STATS) == (False, "")


@pytest.mark.req("AUTO-03")
def test_without_training_statistics_drift_cannot_be_judged():
    assert AutonomyGuard().drift(frame([1000] * 5, 3.0), None) == (False, "")
