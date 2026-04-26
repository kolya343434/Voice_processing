from __future__ import annotations

from pathlib import Path

import numpy as np

from audio_io import save_wav


def _shape_with_formants(y: np.ndarray, sr: int, formants: list[tuple[float, float]]) -> np.ndarray:
    """
    Very simple spectral-envelope shaping using gaussian peaks in frequency domain.
    formants: list of (center_hz, bandwidth_hz).
    """
    n = y.shape[0]
    n_fft = int(2 ** int(np.ceil(np.log2(max(2048, n)))))
    Y = np.fft.rfft(y.astype(np.float64), n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    env = np.ones_like(freqs, dtype=np.float64) * 0.2
    for fc, bw in formants:
        env += np.exp(-0.5 * ((freqs - fc) / max(bw, 1.0)) ** 2)
    Y2 = Y * env
    out = np.fft.irfft(Y2, n=n_fft)[:n]
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.9
    return out.astype(np.float32)


def _harmonic_source(sr: int, dur: float, f0_start: float, f0_end: float, *, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    f0 = np.linspace(f0_start, f0_end, num=n, dtype=np.float64)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    # Harmonic-rich source + slight noise.
    y = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase) + 0.15 * np.sin(4 * phase)
    y += 0.01 * rng.normal(size=n)
    env = (1.0 - np.exp(-30 * t)) * np.exp(-0.3 * t)
    y *= env
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    return y.astype(np.float32)


def _bark_like(sr: int, dur: float, *, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    y = rng.normal(size=n).astype(np.float32) * 0.15
    # bursts
    for center in [0.25, 0.6, 1.0]:
        c = int(center * sr)
        w = int(0.08 * sr)
        if c + w < n:
            env = np.hanning(w).astype(np.float32)
            y[c : c + w] += env * 0.8 * np.sin(2 * np.pi * 180 * np.arange(w) / sr).astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    return y


def generate_demo_variant1(samples_dir: str | Path, *, sr: int = 22050) -> None:
    base = Path(samples_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Typical (rough) Russian vowel formants.
    formants_A = [(700.0, 120.0), (1100.0, 150.0), (2450.0, 200.0)]
    formants_I = [(300.0, 90.0), (2200.0, 180.0), (3000.0, 220.0)]

    src_A = _harmonic_source(sr, dur=6.0, f0_start=90.0, f0_end=520.0, seed=1)
    src_I = _harmonic_source(sr, dur=6.0, f0_start=110.0, f0_end=610.0, seed=2)
    # Keep a portion of the raw harmonic source so F0 is clearly visible for pitch estimation.
    a = 0.35 * src_A + 0.65 * _shape_with_formants(src_A, sr, formants_A)
    i = 0.35 * src_I + 0.65 * _shape_with_formants(src_I, sr, formants_I)
    a = (a / (np.max(np.abs(a)) + 1e-9) * 0.9).astype(np.float32)
    i = (i / (np.max(np.abs(i)) + 1e-9) * 0.9).astype(np.float32)
    other = _bark_like(sr, dur=1.4)

    save_wav(base / "A.wav", a, sr)
    save_wav(base / "I.wav", i, sr)
    save_wav(base / "other.wav", other, sr)
