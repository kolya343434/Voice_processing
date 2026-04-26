from __future__ import annotations

from pathlib import Path

import numpy as np

from audio_io import save_wav


def generate_demo_samples(samples_dir: str | Path, phones: list[str], *, sr: int = 22050) -> None:
    """
    Creates synthetic (not real voice) phone samples so the repo is runnable without a mic.
    Each phone -> short tone burst with different frequency.
    """
    base = Path(samples_dir)
    base.mkdir(parents=True, exist_ok=True)

    freqs = np.linspace(220.0, 880.0, num=max(1, len(phones)), dtype=np.float64)
    for ph, f in zip(phones, freqs, strict=False):
        dur = 0.18 if len(ph) <= 2 else 0.22
        n = int(sr * dur)
        t = np.arange(n, dtype=np.float64) / sr
        env = (1.0 - np.exp(-40.0 * t)) * np.exp(-8.0 * t)
        y = (0.6 * np.sin(2 * np.pi * f * t) + 0.2 * np.sin(2 * np.pi * 2 * f * t)) * env
        y = np.clip(y, -1.0, 1.0).astype(np.float32)
        save_wav(base / f"{ph}.wav", y, sr)
