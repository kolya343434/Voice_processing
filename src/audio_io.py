from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio_mono(path: str | Path, *, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=True)
    y = y.astype(np.float32)
    y = y.mean(axis=1)  # stereo -> mono
    if target_sr is not None and int(target_sr) != int(sr):
        # Linear resample (good enough for lab; avoids extra deps).
        x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        n_new = int(round(len(y) * (target_sr / sr)))
        x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        sr = int(target_sr)
    return y, int(sr)


def save_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), y.astype(np.float32), int(sr))
