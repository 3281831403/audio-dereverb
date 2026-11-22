"""Contract tests for the offline dereverberation exercise.

The tests are intentionally immutable: they encode the required numerical
behaviour and must not be relaxed, skipped, or deleted to make an
implementation pass.
"""

import math

import pytest

from dereverb.suppress import (
    decay_estimate,
    dereverb_batch,
    reconstruct,
    smooth_gain,
    spectral_subtract,
    stft_mag,
    validate,
    wiener_gain,
)


N_FFT = 64
HOP = 16
N_SAMPLES = 512
CONFIG = {
    "n_fft": N_FFT,
    "hop": HOP,
    "k": 3,
    "over_sub": 1.5,
    "floor": 0.05,
    "alpha": 0.6,
}


def _periodic_hann(n_fft):
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * n / n_fft) for n in range(n_fft)]


def _assert_close(actual, expected, tol=1e-9):
    assert isinstance(actual, list)
    assert isinstance(expected, list)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        if isinstance(want, list):
            _assert_close(got, want, tol)
        else:
            assert abs(float(got) - float(want)) <= tol


def _reference_mag_phase(samples, n_fft, hop):
    """Independent reference STFT used only where reconstruct is exercised."""

    window = _periodic_hann(n_fft)
    bins = n_fft // 2 + 1
    frames = 1 + (len(samples) - n_fft) // hop
    mag = []
    phase = []
    for t in range(frames):
        start = t * hop
        mag_row = []
        phase_row = []
        for k in range(bins):
            real = 0.0
            imag = 0.0
            for n in range(n_fft):
                value = samples[start + n] * window[n]
                angle = 2.0 * math.pi * k * n / n_fft
                real += value * math.cos(angle)
                imag -= value * math.sin(angle)
            mag_row.append(math.hypot(real, imag))
            phase_row.append(math.atan2(imag, real))
        mag.append(mag_row)
        phase.append(phase_row)
    return mag, phase


def _rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values))


def _decaying_track():
    return [
        0.8 * math.exp(-n / 100.0) * math.sin(2.0 * math.pi * 5.0 * n / N_FFT)
        for n in range(N_SAMPLES)
    ]


def _direct_plus_tail():
    direct = [0.0] * N_SAMPLES
    for n in range(120, 136):
        envelope = 0.5 - 0.5 * math.cos(2.0 * math.pi * (n - 120) / 16.0)
        direct[n] = math.sin(2.0 * math.pi * 5.0 * n / N_FFT) * envelope

    late = [0.0] * N_SAMPLES
    for n in range(160, N_SAMPLES):
        late[n] = (
            0.25
            * math.exp(-(n - 160) / 80.0)
            * math.sin(2.0 * math.pi * 7.0 * n / N_FFT)
        )
    return [d + r for d, r in zip(direct, late)]


def test_decay_estimate_uses_backward_slope():
    mag = [
        [8.0, 2.0],
        [4.0, 2.0],
        [2.0, 2.0],
        [1.0, 2.0],
    ]
    late = decay_estimate(mag, 3, 3)
    assert late == pytest.approx([0.875, 0.0], abs=1e-9, rel=0.0)
    _assert_close(decay_estimate(mag, 0, 3), [0.0, 0.0])


def test_spectral_subtract_keeps_positive_floor():
    out = spectral_subtract([[1.0, 0.4]], [[2.0, 0.0]], 1.5)
    _assert_close(out, [[1e-12, 0.4]], tol=1e-15)
    assert out[0][0] > 0.0
    assert all(value > 0.0 for row in out for value in row)


def test_wiener_gain_denominator_includes_late():
    gain = wiener_gain([[3.0, 0.0]], [[4.0, 0.0]], 0.0)
    _assert_close(gain, [[0.36, 1.0]])


def test_smooth_gain_smooths_across_frequency():
    gain = [[0.2, 0.2, 1.0, 0.2, 0.2]]
    smoothed = smooth_gain(gain, 0.0)
    _assert_close(smoothed, [[0.2, 0.4, 0.6, 0.4, 0.2]])


def test_reconstruct_honours_hop():
    mag = [[0.0] * 17 for _ in range(3)]
    phase = [[0.0] * 17 for _ in range(3)]
    out = reconstruct(mag, phase, 32, 5)
    assert len(out) == (3 - 1) * 5 + 32


def test_reconstruct_normalizes_overlap():
    n_fft = 64
    hop = 16
    count = 256
    samples = [
        0.4
        * math.sin(2.0 * math.pi * 5.0 * n / (count - 1))
        * math.sin(math.pi * n / (count - 1))
        for n in range(count)
    ]
    mag, phase = _reference_mag_phase(samples, n_fft, hop)
    out = reconstruct(mag, phase, n_fft, hop)
    assert len(out) == len(samples)
    assert max(abs(got - want) for got, want in zip(out, samples)) < 1e-9


def test_validate_rejects_gain_above_one():
    assert validate([[1.0, 1.0]], [[1.0, 1.2]]) == ["gain>1"]
    assert validate([[1.0]], [[1.0]]) == []


def test_stft_uses_periodic_hann_window():
    n_fft = 8
    hop = 4
    samples = [0.0, 1.0, 0.5, -0.25, 0.75, -0.5, 0.25, 1.0]
    window = _periodic_hann(n_fft)
    expected = []
    for k in range(n_fft // 2 + 1):
        real = 0.0
        imag = 0.0
        for n in range(n_fft):
            value = samples[n] * window[n]
            angle = 2.0 * math.pi * k * n / n_fft
            real += value * math.cos(angle)
            imag -= value * math.sin(angle)
        expected.append(math.hypot(real, imag))

    actual = stft_mag(samples, n_fft, hop)
    assert len(actual) == 1
    _assert_close(actual[0], expected)


def test_batch_does_not_reuse_late_estimate():
    loud_decay = _decaying_track()
    impulse = [0.0] * N_SAMPLES
    impulse[256] = 0.2

    both = dereverb_batch(
        [
            {"name": "loud", "samples": loud_decay},
            {"name": "second", "samples": impulse},
        ],
        CONFIG,
    )
    solo = dereverb_batch([{"name": "second", "samples": impulse}], CONFIG)

    assert set(both) == {"loud", "second"}
    _assert_close(both["second"], solo["second"], tol=1e-12)
    assert max(abs(got - want) for got, want in zip(both["second"], impulse)) < 1e-6


def test_smooth_gain_clamps_alpha():
    gain = [[0.8, 0.8, 0.8], [0.2, 0.2, 0.2]]
    high_alpha = smooth_gain(gain, 2.5)
    _assert_close(high_alpha, [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]])
    assert all(0.0 <= value <= 1.0 for row in high_alpha for value in row)

    low_alpha = smooth_gain([[0.2, 0.2], [0.8, 0.8]], -2.0)
    _assert_close(low_alpha, [[0.2, 0.2], [0.8, 0.8]])


def test_pure_direct_signal_is_not_distorted():
    samples = [0.0] * N_SAMPLES
    samples[256] = 0.7
    out = dereverb_batch([{"name": "direct", "samples": samples}], CONFIG)["direct"]
    assert len(out) == len(samples)
    assert max(abs(got - want) for got, want in zip(out, samples)) < 1e-6


def test_direct_to_reverb_ratio_improves():
    mixed = _direct_plus_tail()
    out = dereverb_batch([{"name": "meeting", "samples": mixed}], CONFIG)["meeting"]

    before = 20.0 * math.log10(_rms(mixed[112:144]) / _rms(mixed[192:480]))
    after = 20.0 * math.log10(_rms(out[112:144]) / _rms(out[192:480]))
    assert after - before > 3.0
    assert _rms(out[192:480]) < 0.8 * _rms(mixed[192:480])


def test_output_length_equals_input_length():
    samples = [
        0.1 * math.sin(2.0 * math.pi * 2.0 * n / N_FFT) for n in range(N_SAMPLES)
    ]
    out = dereverb_batch([{"name": "length", "samples": samples}], CONFIG)["length"]
    assert len(out) == len(samples)


def test_gains_stay_within_unit_interval():
    gain = wiener_gain([[0.0, 1.0, 5.0]], [[0.0, 0.0, 100.0]], 0.1)
    _assert_close(gain, [[1.0, 1.0, 0.1]])
    smoothed = smooth_gain([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]], 0.75)
    for row in gain + smoothed:
        assert all(0.0 <= value <= 1.0 for value in row)


def test_batch_tracks_are_independent():
    first = _decaying_track()
    second = [
        0.15 * math.sin(2.0 * math.pi * 3.0 * n / N_FFT) for n in range(N_SAMPLES)
    ]
    both = dereverb_batch(
        [
            {"name": "first", "samples": first},
            {"name": "second", "samples": second},
        ],
        CONFIG,
    )
    solo_first = dereverb_batch([{"name": "first", "samples": first}], CONFIG)
    solo_second = dereverb_batch([{"name": "second", "samples": second}], CONFIG)

    _assert_close(both["first"], solo_first["first"], tol=1e-12)
    _assert_close(both["second"], solo_second["second"], tol=1e-12)
