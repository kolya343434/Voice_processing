from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import stft, windows  # noqa: E402


def save_spectrogram(
    out_path: str | Path,
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int,
    hop_length: int,
    title: str,
    log_freq: bool = True,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    win = windows.hann(int(n_fft), sym=False)
    f, t, z = stft(
        y.astype(np.float32),
        fs=sr,
        window=win,
        nperseg=int(n_fft),
        noverlap=int(n_fft) - int(hop_length),
        boundary="zeros",
        padded=True,
    )
    mag = np.abs(z).astype(np.float64)
    db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    mesh = ax.pcolormesh(t, f, db, shading="auto", cmap="gray")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if log_freq:
        ax.set_yscale("log")
        ax.set_ylim(max(50.0, float(f[1]) if f.size > 1 else 50.0), float(f.max()))
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
