import numpy as np

from python_Signal_Pipeline import LivePulsePipeline


def test_pulse_windows_are_read_in_microseconds():
    pipeline = LivePulsePipeline(sampling_rate_hz=1_000_000, pulse_on_us=10.0, pulse_off_us=30.0)

    assert pipeline.pulse_on_samples == 10
    assert pipeline.pulse_off_samples == 30


def test_a_pulse_train_yields_one_normalised_vector_per_pulse():
    pipeline = LivePulsePipeline(sampling_rate_hz=1_000_000, pulse_on_us=10.0, pulse_off_us=10.0)
    one_pulse = np.concatenate([np.full(10, 1.0), np.zeros(10)])

    features = pipeline.process_chunk(np.tile(one_pulse, 30))

    assert len(features) == 30
    assert all(vector.shape == (len(pipeline.feature_names),) for vector in features)


def test_a_partial_pulse_is_carried_into_the_next_chunk():
    pipeline = LivePulsePipeline(sampling_rate_hz=1_000_000, pulse_on_us=10.0, pulse_off_us=10.0)
    one_pulse = np.concatenate([np.full(10, 1.0), np.zeros(10)])

    assert len(pipeline.process_chunk(one_pulse[:15])) == 0
    assert len(pipeline.process_chunk(one_pulse[15:])) == 1
