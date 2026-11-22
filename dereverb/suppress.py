"""Dereverberation primitives for the offline audio exercise.

All spectra in this module are one-sided, linear-magnitude spectra produced
with a periodic Hann analysis window.  The only metric expressed in dB is the
direct-to-reverberation ratio, defined as ``20 * log10(rms(direct) / rms(late))``.
Numerical assertions use an absolute tolerance of ``1e-9`` unless a test says
otherwise.  The implementation must use only the Python standard library
(``math``) and pytest; it must be offline, deterministic, and must not sleep.

STFT conventions
----------------
* ``n_fft`` is an even positive frame length; ``hop`` is a positive step.
* Frames start at sample offsets ``0, hop, 2 * hop, ...`` and only complete
  frames are used, so ``T = 1 + (len(samples) - n_fft) // hop``.
* The window is the **periodic** Hann window
  ``w[n] = 0.5 - 0.5 * cos(2 * pi * n / n_fft)``.  It is not the symmetric
  Hann window, and it is not scaled by ``2 / n_fft``.
* Bin ``k`` covers ``0 .. n_fft // 2`` (``n_fft // 2 + 1`` bins).
* A reconstructed signal has length ``(T - 1) * hop + n_fft``.  For an input
  whose length is aligned to a whole number of hops, this is exactly the input
  length.
"""

from __future__ import annotations

import math


MAG_FLOOR = 1e-12
TOL = 1e-9
_LOG_FLOOR = 1e-300
_OLA_FLOOR = 1e-12


def _periodic_hann(n_fft: int) -> list[float]:
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * n / n_fft) for n in range(n_fft)]


def _check_stft_params(n_fft: int, hop: int) -> None:
    if not isinstance(n_fft, int) or isinstance(n_fft, bool) or n_fft < 2 or n_fft % 2 != 0:
        raise ValueError("n_fft must be a positive even integer")
    if not isinstance(hop, int) or isinstance(hop, bool) or hop <= 0:
        raise ValueError("hop must be a positive integer")


def _check_rectangular(name: str, arr) -> tuple[int, int]:
    if not isinstance(arr, list):
        raise ValueError(f"{name} must be a list of rows")
    rows = len(arr)
    cols = None
    for row in arr:
        if not isinstance(row, list):
            raise ValueError(f"{name} must be a list of rows")
        if cols is None:
            cols = len(row)
        elif len(row) != cols:
            raise ValueError(f"{name} must be rectangular")
    return rows, (cols if cols is not None else 0)


def _check_same_shape(mag, late) -> tuple[int, int]:
    rows_a, cols_a = _check_rectangular("mag", mag)
    rows_b, cols_b = _check_rectangular("late", late)
    if rows_a != rows_b or cols_a != cols_b:
        raise ValueError("shapes must match")
    return rows_a, cols_a


def _check_finite_nonnegative(name: str, arr) -> None:
    for row in arr:
        for value in row:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} values must be real numbers")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} values must be finite and non-negative")


def stft_mag(samples: list[float], n_fft: int, hop: int) -> list[list[float]]:
    """Return the one-sided magnitude STFT of ``samples``.

    Parameters
    ----------
    samples:
        Real-valued input.  ``len(samples) >= n_fft`` is required.
    n_fft:
        Even frame length (at least 2).
    hop:
        Positive frame hop.  Frames start at ``0, hop, 2 * hop, ...``; only
        complete frames are emitted, so frame ``t`` covers
        ``samples[t * hop : t * hop + n_fft]``.

    Returns
    -------
    list[list[float]]
        ``T`` rows and ``n_fft // 2 + 1`` columns, where
        ``T = 1 + (len(samples) - n_fft) // hop``.  Element ``[t][k]`` is
        ``abs(sum_n x[t * hop + n] * w[n] * exp(-j * 2*pi*k*n / n_fft))``.
        ``w`` is the **periodic** Hann window
        ``0.5 - 0.5 * cos(2*pi*n/n_fft)``; no ``2 / n_fft`` normalisation is
        applied.

    Raises
    ------
    ValueError
        If ``n_fft`` is not a positive even integer, if ``hop`` is not a
        positive integer, or if ``samples`` is shorter than ``n_fft``.

    Comparison against a reference implementation uses ``abs=1e-9``.
    """
    _check_stft_params(n_fft, hop)
    if not isinstance(samples, list) or len(samples) < n_fft:
        raise ValueError("samples must be a list with len(samples) >= n_fft")
    for value in samples:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("samples must be finite real numbers")

    window = _periodic_hann(n_fft)
    bins = n_fft // 2 + 1
    frames = 1 + (len(samples) - n_fft) // hop
    out = []
    for t in range(frames):
        start = t * hop
        row = []
        for k in range(bins):
            real = 0.0
            imag = 0.0
            for n in range(n_fft):
                value = samples[start + n] * window[n]
                angle = 2.0 * math.pi * k * n / n_fft
                real += value * math.cos(angle)
                imag -= value * math.sin(angle)
            row.append(math.hypot(real, imag))
        out.append(row)
    return out


def _stft_mag_phase(samples: list[float], n_fft: int, hop: int):
    """Complex STFT shared by the batch pipeline (same conventions as stft_mag)."""
    _check_stft_params(n_fft, hop)
    if not isinstance(samples, list) or len(samples) < n_fft:
        raise ValueError("samples must be a list with len(samples) >= n_fft")
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


def decay_estimate(mag: list[list[float]], frame: int, k: int) -> list[float]:
    """Estimate a late-reverberation envelope from the previous ``k`` frames.

    For every frequency bin ``b``, let ``start = max(0, frame - k)`` and fit a
    least-squares straight line to ``ln(max(mag[t][b], 1e-300))`` over the
    inclusive frame indices ``t = start .. frame``.  If the fitted slope is
    ``s``, define ``rate = max(0.0, -s)`` (positive for a decaying envelope).
    The estimate is::

        late[b] = mag[frame][b] * (1 - exp(-rate * (frame - start)))

    Thus a flat or rising bin has ``rate == 0`` and a late envelope of zero,
    while a decaying bin contributes the fraction of its current magnitude
    that is explained by the observed decay.  This deliberately uses the
    backward slope rather than the maximum of the current frame, which would
    keep the late envelope too high.

    Returns one row with the same number of bins as ``mag``.  Return all zeros
    when ``frame <= 0``, ``k <= 0``, fewer than two frames are available, the
    current magnitude is non-positive, or the fit has zero time variance.
    Magnitudes are linear; comparisons use ``abs=1e-9``.
    """
    if not isinstance(mag, list) or len(mag) == 0:
        return []
    if not isinstance(frame, int) or frame < 0 or frame >= len(mag):
        raise ValueError("frame must be a valid row index of mag")
    bins = len(mag[frame])
    if frame <= 0 or k <= 0:
        return [0.0] * bins

    start = max(0, frame - k)
    count = frame - start + 1
    if count < 2:
        return [0.0] * bins

    # Least-squares slope of ln(mag) over t = start .. frame, per bin.
    t_mean = (start + frame) / 2.0
    var_t = 0.0
    for t in range(start, frame + 1):
        var_t += (t - t_mean) ** 2
    if var_t <= 0.0:
        return [0.0] * bins

    span = frame - start
    late = []
    for b in range(bins):
        current = mag[frame][b]
        if current <= 0.0:
            late.append(0.0)
            continue
        y_mean = 0.0
        for t in range(start, frame + 1):
            y_mean += math.log(max(mag[t][b], _LOG_FLOOR))
        y_mean /= count
        cov = 0.0
        for t in range(start, frame + 1):
            cov += (t - t_mean) * (math.log(max(mag[t][b], _LOG_FLOOR)) - y_mean)
        slope = cov / var_t
        rate = max(0.0, -slope)
        late.append(current * (1.0 - math.exp(-rate * span)))
    return late


def spectral_subtract(
    mag: list[list[float]], late: list[list[float]], over_sub: float
) -> list[list[float]]:
    """Perform magnitude spectral subtraction with a positive floor.

    Elementwise::

        out[t][b] = max(mag[t][b] - over_sub * late[t][b], 1e-12)

    The floor is the module constant ``MAG_FLOOR == 1e-12`` and is intentional:
    simply clipping negative results to zero creates spectral holes and audible
    artefacts.  A bin that would become non-positive must therefore be exactly
    ``1e-12``, never ``0.0``.

    ``mag`` and ``late`` must have identical rectangular shapes, all values
    must be finite and non-negative, and ``over_sub >= 0``.  Raise ``ValueError``
    otherwise.  Returns a new list with the same shape; comparisons use
    ``abs=1e-9``.
    """
    rows, cols = _check_same_shape(mag, late)
    _check_finite_nonnegative("mag", mag)
    _check_finite_nonnegative("late", late)
    if not isinstance(over_sub, (int, float)) or isinstance(over_sub, bool):
        raise ValueError("over_sub must be a real number")
    if not math.isfinite(over_sub) or over_sub < 0:
        raise ValueError("over_sub must be finite and >= 0")

    return [
        [max(mag[t][b] - over_sub * late[t][b], MAG_FLOOR) for b in range(cols)]
        for t in range(rows)
    ]


def wiener_gain(
    mag: list[list[float]], late: list[list[float]], floor: float
) -> list[list[float]]:
    """Return a power-domain Wiener gain in ``[floor, 1]``.

    For every time-frequency cell::

        denominator = mag[t][b] ** 2 + late[t][b] ** 2
        gain = 1.0 if denominator <= 0 else mag[t][b] ** 2 / denominator

    The ``late`` term is part of the denominator; omitting it makes the gain
    stay close to one and defeats suppression.  Clamp the result to
    ``[floor, 1.0]``.  ``floor`` must lie in ``[0, 1]``; shapes must match and
    input values must be finite and non-negative.  Raise ``ValueError`` for
    invalid input.

    Returns a new array with the same shape.  Gain comparisons use
    ``abs=1e-9``.
    """
    rows, cols = _check_same_shape(mag, late)
    _check_finite_nonnegative("mag", mag)
    _check_finite_nonnegative("late", late)
    if not isinstance(floor, (int, float)) or isinstance(floor, bool):
        raise ValueError("floor must be a real number")
    if not math.isfinite(floor) or floor < 0.0 or floor > 1.0:
        raise ValueError("floor must lie in [0, 1]")

    out = []
    for t in range(rows):
        row = []
        for b in range(cols):
            m2 = mag[t][b] ** 2
            denominator = m2 + late[t][b] ** 2
            gain = 1.0 if denominator <= 0.0 else m2 / denominator
            row.append(min(1.0, max(floor, gain)))
        out.append(row)
    return out


def smooth_gain(gain: list[list[float]], alpha: float) -> list[list[float]]:
    """Smooth a gain surface across frequency and then across time.

    ``alpha`` is clamped to ``[0, 1]`` before use; values outside that range
    must not be allowed to make the recursive filter diverge.  For each frame
    row, first apply the fixed three-tap frequency smoother ``[0.25, 0.5,
    0.25]`` using edge replication (index ``-1`` clamps to the first bin and
    index ``bins`` clamps to the last bin).  Then apply causal first-order time
    smoothing::

        out[0] = frequency_smoothed[0]
        out[t] = alpha * out[t - 1] + (1 - alpha) * frequency_smoothed[t]

    for ``t >= 1``.  Finally clamp every output to ``[0, 1]``.  Smoothing only
    in time leaves isolated single-bin spikes untouched; smoothing only in
    frequency leaves frame-to-frame flicker.  Both axes are required.
    Comparisons use ``abs=1e-9``.
    """
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha):
        raise ValueError("alpha must be a finite real number")
    alpha = min(1.0, max(0.0, alpha))

    if not isinstance(gain, list):
        raise ValueError("gain must be a list of rows")
    rows = len(gain)
    if rows == 0:
        return []
    cols = len(gain[0])

    # Frequency smoothing with edge replication.
    freq_smoothed = []
    for row in gain:
        if len(row) != cols:
            raise ValueError("gain must be rectangular")
        smoothed_row = []
        for b in range(cols):
            left = row[b - 1] if b - 1 >= 0 else row[0]
            right = row[b + 1] if b + 1 < cols else row[cols - 1]
            smoothed_row.append(0.25 * left + 0.5 * row[b] + 0.25 * right)
        freq_smoothed.append(smoothed_row)

    # Causal first-order time smoothing, then clamp to [0, 1].
    out = [freq_smoothed[0]]
    for t in range(1, rows):
        prev = out[t - 1]
        cur = freq_smoothed[t]
        out.append([alpha * prev[b] + (1.0 - alpha) * cur[b] for b in range(cols)])
    return [[min(1.0, max(0.0, value)) for value in row] for row in out]


def reconstruct(
    mag: list[list[float]],
    phase: list[list[float]],
    n_fft: int,
    hop: int,
) -> list[float]:
    """Invert a one-sided STFT back to a real signal.

    Build the Hermitian ``n_fft``-point spectrum for each frame from ``mag`` and
    ``phase`` (DC and Nyquist are single bins; the other bins are mirrored with
    conjugates), take the inverse DFT, multiply by the same periodic Hann
    synthesis window, overlap-add the frames at offsets ``0, hop, 2 * hop,
    ...``, and divide every sample by the sum of squared overlapping windows::

        out[n] = sum_t w[n - t*hop] * y_t[n - t*hop]
                 / sum_t w[n - t*hop] ** 2

    where ``y_t`` is the inverse DFT of frame ``t``.  The window normalisation
    is not optional: omitting it makes the amplitude ripple in overlapped
    regions.  When the denominator of a sample is at most ``1e-12``, output
    ``0.0`` for that sample.  The hop passed here must be used for synthesis;
    substituting a hard-coded hop changes the output length.

    The output length is exactly ``(len(mag) - 1) * hop + n_fft``.  Shape
    mismatches, invalid FFT parameters, or non-finite values raise
    ``ValueError``.  Round-trip comparisons use ``abs=1e-9``.
    """
    _check_stft_params(n_fft, hop)
    rows, cols = _check_same_shape(mag, phase)
    if rows == 0:
        raise ValueError("mag and phase must contain at least one frame")
    if cols != n_fft // 2 + 1:
        raise ValueError("bin count must equal n_fft // 2 + 1")
    for name, arr in (("mag", mag), ("phase", phase)):
        for row in arr:
            for value in row:
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"{name} values must be finite real numbers")

    window = _periodic_hann(n_fft)
    length = (rows - 1) * hop + n_fft
    out = [0.0] * length
    norm = [0.0] * length
    half = n_fft // 2

    for t in range(rows):
        mag_row = mag[t]
        phase_row = phase[t]
        offset = t * hop
        for n in range(n_fft):
            # Inverse DFT of the Hermitian spectrum at time index n.
            total = mag_row[0] * math.cos(phase_row[0])
            for k in range(1, half):
                angle = phase_row[k] + 2.0 * math.pi * k * n / n_fft
                total += 2.0 * mag_row[k] * math.cos(angle)
            total += mag_row[half] * math.cos(phase_row[half] + math.pi * n)
            y = total / n_fft
            w = window[n]
            out[offset + n] += w * y
            norm[offset + n] += w * w

    for i in range(length):
        if norm[i] <= _OLA_FLOOR:
            out[i] = 0.0
        else:
            out[i] /= norm[i]
    return out


def validate(mag: list[list[float]], gain: list[list[float]]) -> list[str]:
    """Validate a magnitude spectrogram and its gain surface.

    Return a list of issue codes, each code at most once, in this order:

    ``"shape"``
        ``mag`` and ``gain`` are not two rectangular arrays with identical row
        and column counts.  In this case return ``["shape"]`` immediately.
    ``"nan"``
        Any value in either array is NaN or infinite.  If present, return
        ``["shape"]`` (if applicable) or ``["nan"]`` without further checks.
    ``"negative"``
        Any magnitude or gain is negative.
    ``"gain>1"``
        Any gain is greater than ``1.0 + 1e-9``.  An over-unity gain is a bug
        even when all magnitudes are valid.

    Valid input returns ``[]``.  Empty arrays are shape-valid only when both
    arrays are empty in the same way; a zero-row array with a non-zero gain is
    a shape error.
    """

    def _shape(arr):
        if not isinstance(arr, list):
            return None
        rows = len(arr)
        cols = None
        for row in arr:
            if not isinstance(row, list):
                return None
            if cols is None:
                cols = len(row)
            elif len(row) != cols:
                return None
        return rows, (cols if cols is not None else 0)

    shape_mag = _shape(mag)
    shape_gain = _shape(gain)
    if shape_mag is None or shape_gain is None or shape_mag != shape_gain:
        return ["shape"]

    for arr in (mag, gain):
        for row in arr:
            for value in row:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return ["nan"]
                if not math.isfinite(value):
                    return ["nan"]

    issues = []
    for arr in (mag, gain):
        for row in arr:
            for value in row:
                if value < 0.0:
                    issues.append("negative")
                    break
            if "negative" in issues:
                break
        if "negative" in issues:
            break

    for row in gain:
        for value in row:
            if value > 1.0 + TOL:
                issues.append("gain>1")
                break
        if "gain>1" in issues:
            break

    return issues


def dereverb_batch(tracks: list[dict], config: dict) -> dict[str, list[float]]:
    """Dereverberate several tracks independently.

    ``tracks`` is a list of dictionaries with a unique string ``"name"`` and a
    real ``"samples"`` list.  ``config`` must contain ``n_fft``, ``hop``,
    ``k``, ``over_sub``, ``floor``, and ``alpha``.  The return value is a dict
    mapping each name (in input order) to its processed samples.

    For every track, independently:

    1. compute the complex STFT with the same periodic Hann window and framing
       as :func:`stft_mag`, obtaining magnitude and phase;
    2. for each frame ``t``, call ``decay_estimate(mag, t, k)`` on **that
       track's** magnitude;
    3. call ``spectral_subtract(mag, late, over_sub)``;
    4. call ``wiener_gain(clean, late, floor)``;
    5. call ``smooth_gain(gain, alpha)``;
    6. multiply ``clean`` by the smoothed gain and call
       ``reconstruct(enhanced_mag, phase, n_fft, hop)``.

    No late envelope, gain, or other per-track state may be reused across
    tracks: putting a second track in the same batch must produce exactly the
    same samples as processing it alone.  The output length follows
    :func:`reconstruct` and equals the input length for an aligned track.

    Duplicate or missing names, missing config keys, malformed tracks, and
    invalid shapes raise ``ValueError``.  This function must not mutate its
    arguments.  Comparisons use ``abs=1e-12`` for batch-versus-solo equality.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    required = ("n_fft", "hop", "k", "over_sub", "floor", "alpha")
    for key in required:
        if key not in config:
            raise ValueError(f"config is missing {key!r}")
    n_fft = config["n_fft"]
    hop = config["hop"]
    k = config["k"]
    over_sub = config["over_sub"]
    floor = config["floor"]
    alpha = config["alpha"]
    _check_stft_params(n_fft, hop)
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("k must be an integer")

    if not isinstance(tracks, list):
        raise ValueError("tracks must be a list")

    result: dict[str, list[float]] = {}
    for track in tracks:
        if not isinstance(track, dict):
            raise ValueError("each track must be a dict")
        name = track.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("each track needs a non-empty string name")
        if name in result:
            raise ValueError(f"duplicate track name {name!r}")
        samples = track.get("samples")
        if not isinstance(samples, list):
            raise ValueError("each track needs a samples list")
        for value in samples:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError("samples must be finite real numbers")

        # All per-track state is computed fresh inside this loop iteration.
        mag, phase = _stft_mag_phase(samples, n_fft, hop)
        late = [decay_estimate(mag, t, k) for t in range(len(mag))]
        clean = spectral_subtract(mag, late, over_sub)
        gain = wiener_gain(clean, late, floor)
        smoothed = smooth_gain(gain, alpha)
        enhanced = [
            [clean[t][b] * smoothed[t][b] for b in range(len(clean[t]))]
            for t in range(len(clean))
        ]
        result[name] = reconstruct(enhanced, phase, n_fft, hop)

    return result
