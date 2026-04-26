from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_io import load_audio_mono


@dataclass(frozen=True)
class SynthConfig:
    sample_rate: int = 22050
    crossfade_ms: int = 20


def _linear_crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.concatenate([a, b])
    n = int(min(n, a.shape[0], b.shape[0]))
    if n <= 0:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, num=n, endpoint=False, dtype=np.float32)
    fade_in = 1.0 - fade_out
    mid = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([a[:-n], mid, b[n:]])


def synth_concat(segments: list[np.ndarray]) -> np.ndarray:
    if not segments:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate([s.astype(np.float32) for s in segments])


def synth_crossfade(segments: list[np.ndarray], *, crossfade_samples: int) -> np.ndarray:
    if not segments:
        return np.zeros((0,), dtype=np.float32)
    y = segments[0].astype(np.float32)
    for seg in segments[1:]:
        y = _linear_crossfade(y, seg.astype(np.float32), crossfade_samples)
    return y


def load_phone_segments(phones: list[str], samples_dir: str | Path, cfg: SynthConfig) -> list[np.ndarray]:
    base = Path(samples_dir)
    segments: list[np.ndarray] = []
    missing: list[str] = []
    for ph in phones:
        wav = base / f"{ph}.wav"
        if not wav.exists():
            missing.append(ph)
            continue
        y, _ = load_audio_mono(wav, target_sr=cfg.sample_rate)
        segments.append(y)
    if missing:
        raise FileNotFoundError(f"Missing samples in {base}: {', '.join(missing)}")
    return segments


def normalize_audio(y: np.ndarray, *, peak: float = 0.98) -> np.ndarray:
    y = y.astype(np.float32)
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m <= 0:
        return y
    return np.clip(y * (peak / m), -1.0, 1.0)
