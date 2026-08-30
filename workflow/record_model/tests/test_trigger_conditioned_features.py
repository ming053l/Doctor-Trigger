import numpy as np

from src.trigger_conditioned import (
    RingSpec,
    compute_delta_s,
    compute_host_score,
    compute_template_score,
    compute_trigger_projection,
    deterministic_wrong_keys,
    hermitian_half_plane_mask,
    recover_host_phase,
    wrap_phase,
)


def test_wrap_phase_range_and_periodicity():
    x = np.linspace(-20 * np.pi, 20 * np.pi, 1000)
    w = wrap_phase(x)
    assert np.all(w >= -np.pi)
    assert np.all(w < np.pi)
    np.testing.assert_allclose(wrap_phase(x + 2 * np.pi), w, atol=1e-10)


def test_delta_s_active_magnitude_and_support():
    ring = RingSpec(shape=(64, 64), bands=(0.1, 0.45), half_plane=True)
    delta = compute_delta_s(123, alpha=0.02, beta=0.15, ring=ring, code_mode="zero_pi")
    active = np.abs(delta.delta[delta.support])
    nonzero = active[active > 1e-8]
    assert np.isfinite(active).all()
    assert nonzero.size > 0
    np.testing.assert_allclose(np.median(nonzero), 0.009423, atol=8e-4)
    assert not np.any(np.logical_and(delta.support, np.logical_not(hermitian_half_plane_mask((64, 64)))))


def test_host_recovery_scores():
    rng = np.random.default_rng(7)
    phi_hat = rng.uniform(-np.pi, np.pi, size=(32, 32))
    ring = RingSpec(shape=(32, 32), bands=(0.1, 0.45), half_plane=True)
    delta = compute_delta_s(99, alpha=0.02, beta=0.15, ring=ring, code_mode="zero_pi")
    template = wrap_phase(phi_hat + delta.delta)
    recovered = recover_host_phase(template, delta)
    np.testing.assert_allclose(recovered[delta.support], phi_hat[delta.support], atol=1e-10)
    h_if_template_query = compute_host_score(template, recovered, delta.support)
    c_if_template_query = compute_template_score(template, template, delta.support)
    h_if_host_query = compute_host_score(recovered, recovered, delta.support)
    assert 0.999 <= h_if_template_query <= 1.0
    assert 0.999999 <= c_if_template_query <= 1.0
    assert 0.999999 <= h_if_host_query <= 1.0


def test_trigger_projection_correct_vs_wrong_key():
    rng = np.random.default_rng(5)
    phi_hat = rng.uniform(-np.pi, np.pi, size=(64, 64))
    ring = RingSpec(shape=(64, 64), bands=(0.1, 0.45), half_plane=True)
    correct = compute_delta_s(1234, alpha=0.02, beta=0.15, ring=ring, code_mode="zero_pi")
    noise = rng.normal(0.0, 0.0005, size=(64, 64))
    phi_j = wrap_phase(phi_hat + correct.delta + noise)
    g_correct = compute_trigger_projection(phi_j, phi_hat, correct)
    wrong = compute_delta_s(deterministic_wrong_keys(1234, n_wrong=1)[0], alpha=0.02, beta=0.15, ring=ring, code_mode="zero_pi")
    g_wrong = compute_trigger_projection(phi_j, phi_hat, wrong)
    assert 0.8 <= g_correct <= 1.2
    assert g_correct - g_wrong > 0.2


def test_zero_trigger_is_safe():
    rng = np.random.default_rng(4)
    phi_hat = rng.uniform(-np.pi, np.pi, size=(32, 32))
    ring = RingSpec(shape=(32, 32), bands=(0.1, 0.45), half_plane=True)
    delta = compute_delta_s(5, alpha=0.0, beta=0.15, ring=ring, code_mode="zero_pi")
    g = compute_trigger_projection(phi_hat, phi_hat, delta)
    assert np.isfinite(g)
    assert g == 0.0


def test_smoke_script_does_not_call_dataset_embedding():
    script = open("scripts/record_model_release_smoke.sh", encoding="utf-8").read()
    forbidden = ["synth_trigger.py", "build_trigger_eval_set.py", "release_generation", "main/fakes/rebuild"]
    assert not any(token in script for token in forbidden)
