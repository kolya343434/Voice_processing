from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter


@dataclass(frozen=True)
class PitchConfig:
    frame_sec: float = 0.04
    hop_sec: float = 0.01
    fmin_hz: float = 70.0
    fmax_hz: float = 1200.0
    voiced_threshold: float = 0.3  # normalized autocorr peak


@dataclass(frozen=True)
class HarmonicsConfig:
    max_harmonic_hz: float = 8000.0
    relative_threshold: float = 0.12


@dataclass(frozen=True)
class FormantsConfig:
    dt_sec: float = 0.1
    df_hz: float = 50.0
    min_hz: float = 150.0
    max_hz: float = 4000.0
    smooth_hz: float = 25.0
    cepstral_lifter_ms: float = 3.5
    top_n: int = 3


def frame_signal(y: np.ndarray, sr: int, *, frame_sec: float, hop_sec: float) -> tuple[np.ndarray, np.ndarray]:
    y = y.astype(np.float32)
    n = y.shape[0]
    frame = int(round(frame_sec * sr))
    hop = int(round(hop_sec * sr))
    frame = max(16, frame)
    hop = max(1, hop)
    n_frames = 1 + max(0, (n - frame) // hop)
    frames = np.zeros((n_frames, frame), dtype=np.float32)
    times = np.zeros((n_frames,), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frames[i] = y[start : start + frame]
        times[i] = (start + frame / 2) / sr
    return frames, times


def estimate_pitch_autocorr(y: np.ndarray, sr: int, cfg: PitchConfig) -> dict[str, object]:
    frames, times = frame_signal(y, sr, frame_sec=cfg.frame_sec, hop_sec=cfg.hop_sec)
    f0 = np.full((frames.shape[0],), np.nan, dtype=np.float64)
    voiced = np.zeros((frames.shape[0],), dtype=bool)

    lag_min = int(math.floor(sr / float(cfg.fmax_hz)))
    lag_max = int(math.ceil(sr / float(cfg.fmin_hz)))
    lag_min = max(lag_min, 1)
    lag_max = max(lag_max, lag_min + 1)

    for i, x in enumerate(frames):
        x = x - float(np.mean(x))
        e = float(np.sum(x * x))
        if e <= 1e-8:
            continue
        # Autocorrelation (biased) via FFT-free method for small frames.
        r = np.correlate(x, x, mode="full")
        r = r[r.size // 2 :]
        r0 = float(r[0])
        if r0 <= 0:
            continue
        r_norm = r / r0
        search = r_norm[lag_min:lag_max]
        if search.size == 0:
            continue
        k = int(np.argmax(search)) + lag_min
        peak = float(r_norm[k])
        if peak >= float(cfg.voiced_threshold):
            f0[i] = sr / k
            voiced[i] = True

    voiced_f0 = f0[voiced]
    out = {
        "times_sec": times,
        "f0_hz": f0,
        "voiced_mask": voiced,
        "f0_min_hz": float(np.nanmin(voiced_f0)) if voiced_f0.size else None,
        "f0_max_hz": float(np.nanmax(voiced_f0)) if voiced_f0.size else None,
        "voiced_ratio": float(np.mean(voiced)) if voiced.size else 0.0,
    }
    return out


def estimate_pitch_cepstrum(y: np.ndarray, sr: int, cfg: PitchConfig, *, n_fft: int = 4096) -> dict[str, object]:
    """
    Cepstrum-based pitch estimation (more robust to formant shaping than spectral peak).
    """
    frames, times = frame_signal(y, sr, frame_sec=cfg.frame_sec, hop_sec=cfg.hop_sec)
    f0 = np.full((frames.shape[0],), np.nan, dtype=np.float64)
    voiced = np.zeros((frames.shape[0],), dtype=bool)

    qmin = 1.0 / float(cfg.fmax_hz)
    qmax = 1.0 / float(cfg.fmin_hz)
    # Use ceil/floor so resulting f0 is within [fmin..fmax].
    kmin = int(math.ceil(qmin * sr))  # smallest quefrency index -> largest f0 <= fmax
    kmax = int(math.floor(qmax * sr))  # largest quefrency index -> smallest f0 >= fmin
    kmin = max(kmin, 1)
    kmax = max(kmax, kmin + 1)

    win = np.hanning(frames.shape[1]).astype(np.float64)

    for i, x in enumerate(frames):
        x = x.astype(np.float64)
        x = (x - float(np.mean(x))) * win
        if float(np.sum(x * x)) <= 1e-8:
            continue

        if x.shape[0] < n_fft:
            xw = np.pad(x, (0, n_fft - x.shape[0]))
        else:
            xw = x[:n_fft]

        spec = np.fft.rfft(xw)
        mag = np.abs(spec)
        log_mag = np.log(np.maximum(mag, 1e-12))
        ceps = np.fft.irfft(log_mag)

        search = ceps[kmin:kmax]
        if search.size == 0:
            continue
        k = int(np.argmax(search)) + kmin
        peak = float(ceps[k])
        base = float(np.median(search))
        spread = float(np.std(search)) + 1e-9
        score = (peak - base) / spread  # peak prominence in std units
        if score >= float(cfg.voiced_threshold):
            f0[i] = sr / k
            voiced[i] = True

    voiced_f0 = f0[voiced]
    return {
        "method": "cepstrum",
        "times_sec": times,
        "f0_hz": f0,
        "voiced_mask": voiced,
        "f0_min_hz": float(np.nanmin(voiced_f0)) if voiced_f0.size else None,
        "f0_max_hz": float(np.nanmax(voiced_f0)) if voiced_f0.size else None,
        "voiced_ratio": float(np.mean(voiced)) if voiced.size else 0.0,
    }


def _spectrum_mag(x: np.ndarray, sr: int, *, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    win = np.hanning(x.shape[0]).astype(np.float64)
    xw = x.astype(np.float64) * win
    if xw.shape[0] < n_fft:
        xw = np.pad(xw, (0, n_fft - xw.shape[0]))
    spec = np.fft.rfft(xw[:n_fft])
    mag = np.abs(spec).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float64)
    return freqs, mag


def most_timbre_colored_f0(
    y: np.ndarray,
    sr: int,
    pitch: dict[str, object],
    cfg: HarmonicsConfig,
    *,
    n_fft: int = 8192,
) -> dict[str, object]:
    times = pitch["times_sec"]
    f0 = pitch["f0_hz"]
    voiced = pitch["voiced_mask"]

    frames, _ = frame_signal(y, sr, frame_sec=0.04, hop_sec=0.01)
    best = {"harmonics": -1, "f0_hz": None, "time_sec": None}

    for i in range(frames.shape[0]):
        if not bool(voiced[i]) or not np.isfinite(float(f0[i])):
            continue
        f0_i = float(f0[i])
        if f0_i <= 0:
            continue
        freqs, mag = _spectrum_mag(frames[i], sr, n_fft=n_fft)
        fmax = float(cfg.max_harmonic_hz)
        h_max = int(math.floor(min(fmax, freqs[-1]) / f0_i))
        if h_max < 2:
            continue

        # Fundamental magnitude near f0 (±1 bin)
        idx0 = int(np.argmin(np.abs(freqs - f0_i)))
        m0 = float(mag[idx0])
        if m0 <= 0:
            continue
        thr = float(cfg.relative_threshold) * m0

        count = 0
        for h in range(1, h_max + 1):
            fh = h * f0_i
            idx = int(np.argmin(np.abs(freqs - fh)))
            if float(mag[idx]) >= thr:
                count += 1
        if count > int(best["harmonics"]):
            best = {"harmonics": int(count), "f0_hz": f0_i, "time_sec": float(times[i])}

    return best


def formants_over_time(
    y: np.ndarray,
    sr: int,
    cfg: FormantsConfig,
    *,
    n_fft: int,
) -> dict[str, object]:
    """
    Task-style formants:
    - time windows with step Δt
    - find peaks of smoothed log-spectrum envelope with minimum separation ~ Δf
    """
    dt = float(cfg.dt_sec)
    hop = max(1, int(round(dt * sr)))

    win_len = int(round(2 * dt * sr))
    win_len = min(max(win_len, 1024), y.shape[0])

    df_bin = sr / n_fft
    min_dist_bins = max(1, int(round(float(cfg.df_hz) / df_bin)))
    lifter_len = int(round((float(cfg.cepstral_lifter_ms) / 1000.0) * sr))
    lifter_len = max(3, min(lifter_len, (n_fft // 2) - 1))

    times: list[float] = []
    formants_all: list[list[float]] = []

    for start in range(0, max(1, y.shape[0] - win_len + 1), hop):
        seg = y[start : start + win_len]
        t_center = (start + win_len / 2) / sr
        # Pre-emphasis improves formant visibility.
        seg = seg.astype(np.float64)
        seg_pe = seg.copy()
        seg_pe[1:] = seg_pe[1:] - 0.97 * seg_pe[:-1]
        freqs, mag = _spectrum_mag(seg_pe.astype(np.float32), sr, n_fft=n_fft)

        fmin = float(cfg.min_hz)
        fmax = float(cfg.max_hz)
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if idx.size < 10:
            continue

        # Cepstral liftering: smooth the log-spectrum to remove harmonics (pitch) and keep envelope.
        log_mag_full = np.log(np.maximum(mag, 1e-12))
        ceps = np.fft.irfft(log_mag_full, n=n_fft)
        if ceps.size > 2 * lifter_len:
            ceps[lifter_len : -lifter_len] = 0.0
        env_log_full = np.fft.rfft(ceps, n=n_fft).real
        env_log = env_log_full[idx]

        pidx, _ = find_peaks(env_log, distance=min_dist_bins)
        if pidx.size == 0:
            continue
        energies = env_log[pidx]
        order = np.argsort(-energies)[: int(cfg.top_n)]
        chosen = sorted(float(freqs[idx[pidx[i]]]) for i in order)
        times.append(float(t_center))
        formants_all.append(chosen)

    return {"dt_sec": cfg.dt_sec, "df_hz": cfg.df_hz, "times_sec": times, "formants_hz": formants_all}
