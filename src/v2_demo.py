from __future__ import annotations

from pathlib import Path

import numpy as np

from audio_io import save_wav


def generate_inventory_demo(samples_dir: str | Path, phones: list[str], *, sr: int = 22050) -> None:
    """
    Creates 63 short synthetic WAV files to make the repo runnable without a mic.
    Replace these files with your real recordings for submission.
    """
    base = Path(samples_dir)
    base.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    freqs = np.linspace(160.0, 1200.0, num=max(1, len(phones)), dtype=np.float64)
    for ph, f in zip(phones, freqs, strict=False):
        # Special tokens: silence/pause.
        if ph in {"sil", "pause", "br", "cl"}:
            dur = 0.12
            y = np.zeros(int(sr * dur), dtype=np.float32)
            save_wav(base / f"{ph}.wav", y, sr)
            continue

        dur = 0.14 + 0.02 * (len(ph) % 3)
        n = int(sr * dur)
        t = np.arange(n, dtype=np.float64) / sr
        env = (1.0 - np.exp(-60.0 * t)) * np.exp(-10.0 * t)
        y = (
            0.55 * np.sin(2 * np.pi * f * t)
            + 0.22 * np.sin(2 * np.pi * 2 * f * t)
            + 0.10 * np.sin(2 * np.pi * 3 * f * t)
        ) * env
        y += 0.005 * rng.normal(size=n)
        y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
        save_wav(base / f"{ph}.wav", y, sr)
