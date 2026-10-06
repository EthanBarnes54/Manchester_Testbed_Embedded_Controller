"""The GRU controller: training, online updates, checkpoints and thread safety."""

import threading
import time

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

import python_RNN_Controller as rnn
from helpers import sweep_shaped_frame


@pytest.fixture
def trained():
    """A model whose scaler has been fitted, so every inference path is live."""

    frame = sweep_shaped_frame()
    rnn.train_model(frame, number_of_epochs=2)
    return frame


def test_learning_rate_reaches_the_optimiser(trained):
    applied = rnn.set_learning_rate(3e-3)

    assert applied == pytest.approx(3e-3)
    assert all(group["lr"] == pytest.approx(3e-3) for group in rnn.optimiser.param_groups)

    rnn.set_learning_rate(rnn.LEARNING_RATE)


def test_online_update_returns_a_triple_on_every_exit(trained, monkeypatch):
    assert len(rnn.online_update(trained.head(5))) == 3  # too little data
    assert len(rnn.online_update(trained)) == 3  # a real step

    monkeypatch.setattr(rnn, "scaler", StandardScaler())
    assert rnn.online_update(trained) == (False, None, None)  # nothing fitted yet


def test_training_reports_held_out_scores(trained):
    metrics = rnn.train_model(trained, number_of_epochs=2)

    assert {"loss", "r2", "validation_r2", "validation_rmse", "validation_mae"} <= metrics.keys()


def test_a_checkpoint_round_trips_weights_and_scaler(trained, tmp_path, monkeypatch):
    monkeypatch.setattr(rnn, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(rnn, "MODEL_PATH", tmp_path / f"{rnn.MODEL_BASENAME}.pt")

    rnn.save_nn_weights(rnn.model, rnn.scaler)
    saved_weights = {name: tensor.clone() for name, tensor in rnn.model.state_dict().items()}
    saved_mean = rnn.scaler.mean_.copy()

    fresh_model = rnn._RNN(rnn.INPUT_SIZE, rnn.HIDDEN_SIZE, rnn.OUTPUT_SIZE)
    fresh_scaler = StandardScaler()

    assert rnn.load_previous_weights(fresh_model, fresh_scaler)
    assert all(torch.equal(saved_weights[name], tensor) for name, tensor in fresh_model.state_dict().items())
    assert np.allclose(fresh_scaler.mean_, saved_mean)


def test_proposals_stay_inside_the_output_range(trained):
    targets = rnn.propose_control_vector(trained, num_candidates=16)

    assert len(targets) == 5
    assert all(0.0 <= volts <= rnn.ANALOG_VREF for volts in targets)


def test_the_shared_model_is_never_used_by_two_threads_at_once(trained, monkeypatch):
    inside, overlaps, errors = set(), [], []
    guard = threading.Lock()
    real_forward = rnn.model.forward

    def probe(*args, **kwargs):
        me = threading.get_ident()

        with guard:
            if inside - {me}:
                overlaps.append(me)
            inside.add(me)

        try:
            time.sleep(0.002)  # widen the window a real forward pass already has
            return real_forward(*args, **kwargs)
        finally:
            with guard:
                inside.discard(me)

    monkeypatch.setattr(rnn.model, "forward", probe)
    stop = threading.Event()

    def hammer(action):
        while not stop.is_set():
            try:
                action()
            except Exception as fault:
                errors.append(repr(fault))

    workers = [
        threading.Thread(target=hammer, args=(lambda: rnn.train_model(trained, number_of_epochs=2),)),
        threading.Thread(target=hammer, args=(lambda: rnn.online_update(trained.tail(200)),)),
        threading.Thread(target=hammer, args=(lambda: rnn.propose_control_vector(trained.tail(50), num_candidates=4),)),
        threading.Thread(target=hammer, args=(lambda: rnn.set_optimiser_type("sgd" if time.time() % 0.2 < 0.1 else "adam"),)),
    ]

    for worker in workers:
        worker.start()

    time.sleep(2.0)
    stop.set()

    for worker in workers:
        worker.join()

    rnn.set_optimiser_type(rnn.OPTIMISER_TYPE)

    assert errors == []
    assert overlaps == []


def test_saliency_scoring_does_not_hold_the_model_lock(trained):
    result = {}
    worker = threading.Thread(target=lambda: result.update(rnn.compute_feature_saliencies(trained, max_samples=40, num_permutations=30)))
    worker.start()
    time.sleep(0.3)

    started = time.time()
    rnn.set_learning_rate(rnn.LEARNING_RATE)
    waited = time.time() - started
    worker.join()

    assert waited < 0.2
    assert len(result["feature_names"]) == 5


def test_pipeline_controller_builds_and_predicts():
    controller = rnn.RNNController(feature_dim=6)
    pins = np.zeros(5)

    predictions = [controller.step_features(pins, np.zeros(6)) for _ in range(rnn.SEQUENCE_LENGTH)]

    assert predictions[:-1] == [None] * (rnn.SEQUENCE_LENGTH - 1)
    assert predictions[-1].shape == (1,)


def test_training_records_the_held_out_score_the_weights_earned(trained):
    metrics = rnn.train_model(trained, number_of_epochs=2)
    validation = rnn.get_validation_metrics()

    assert validation["validation_r2"] == pytest.approx(metrics["validation_r2"])
    assert validation["online_updates_since"] == 0

    rnn.online_update(trained)
    assert rnn.get_validation_metrics()["online_updates_since"] == 1


def test_the_held_out_score_travels_with_the_checkpoint(trained, tmp_path, monkeypatch):
    monkeypatch.setattr(rnn, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(rnn, "MODEL_PATH", tmp_path / f"{rnn.MODEL_BASENAME}.pt")
    rnn.train_model(trained, number_of_epochs=2)
    saved = rnn.get_validation_metrics()
    rnn.save_nn_weights(rnn.model, rnn.scaler)

    rnn.VALIDATION.clear()
    assert rnn.load_previous_weights(rnn._RNN(rnn.INPUT_SIZE, rnn.HIDDEN_SIZE, rnn.OUTPUT_SIZE), StandardScaler())
    assert rnn.get_validation_metrics() == saved


def test_the_training_distribution_is_available_for_drift_checks(trained):
    stats = rnn.get_training_feature_stats()

    assert stats["columns"] == ["pin_1", "pin_2", "pin_3", "pin_4", "pin_5", "voltage"]
    assert len(stats["mean"]) == len(stats["scale"]) == 6 and all(scale > 0 for scale in stats["scale"])
