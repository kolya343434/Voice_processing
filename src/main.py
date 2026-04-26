from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_io import load_audio_mono
from plotting import save_spectrogram
from voice_analysis import (
    FormantsConfig,
    HarmonicsConfig,
    PitchConfig,
    estimate_pitch_cepstrum,
    formants_over_time,
    most_timbre_colored_f0,
)
from voice_demo import generate_demo_variant1


def _read_cfg(repo_root: Path) -> dict:
    cfg_path = repo_root / "config" / "variant1.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _read_cfg(repo_root)
    if int(cfg.get("variant", 1)) != 1:
        raise SystemExit("This repo is prepared for lab 10, variant 1.")

    parser = argparse.ArgumentParser(description="Lab 10 (variant 1): voice range, timbre, formants.")
    parser.add_argument("--A", dest="a_path", type=str, default=None, help="Path to vowel A wav.")
    parser.add_argument("--I", dest="i_path", type=str, default=None, help="Path to vowel I wav.")
    parser.add_argument("--other", dest="o_path", type=str, default=None, help="Path to bark/meow/etc wav.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo wavs and run analysis.")
    parser.add_argument("--full", action="store_true", help="Include full arrays in JSON (otherwise summary only).")
    parser.add_argument("--json-out", type=str, default=None, help="Write report JSON to file.")
    args = parser.parse_args()

    sr = int(cfg["sample_rate"])
    spec_cfg = cfg["spectrogram"]
    pitch_cfg = PitchConfig(**cfg["pitch"])
    harm_cfg = HarmonicsConfig(**cfg["harmonics"])
    form_cfg = FormantsConfig(**cfg["formants"])

    samples_dir = repo_root / "samples"
    assets_dir = repo_root / "assets"

    if args.demo:
        generate_demo_variant1(samples_dir, sr=sr)

    a_path = Path(args.a_path) if args.a_path else repo_root / cfg["inputs"]["A_wav"]
    i_path = Path(args.i_path) if args.i_path else repo_root / cfg["inputs"]["I_wav"]
    o_path = Path(args.o_path) if args.o_path else repo_root / cfg["inputs"]["other_wav"]

    report: dict[str, object] = {
        "lab": 10,
        "variant": 1,
        "sample_rate": sr,
        "inputs": {"A": str(a_path), "I": str(i_path), "other": str(o_path)},
        "spectrogram_png": {
            "A": str(assets_dir / "spectrogram_A.png"),
            "I": str(assets_dir / "spectrogram_I.png"),
            "other": str(assets_dir / "spectrogram_other.png"),
        },
    }

    def analyze_one(name: str, path: Path) -> dict[str, object]:
        y, _ = load_audio_mono(path, target_sr=sr)
        save_spectrogram(
            assets_dir / f"spectrogram_{name}.png",
            y,
            sr,
            n_fft=int(spec_cfg["n_fft"]),
            hop_length=int(spec_cfg["hop_length"]),
            title=f"Spectrogram ({name}, Hann, log-freq)",
            log_freq=True,
        )
        pitch = estimate_pitch_cepstrum(y, sr, pitch_cfg, n_fft=int(spec_cfg["n_fft"]))
        best = most_timbre_colored_f0(y, sr, pitch, harm_cfg)
        form = formants_over_time(y, sr, form_cfg, n_fft=int(spec_cfg["n_fft"]))

        # Summaries for README-friendly report
        form_list = form.get("formants_hz", [])
        flat = [f for row in form_list for f in row]
        form_mean = float(sum(flat) / len(flat)) if flat else None

        out: dict[str, object] = {
            "duration_sec": float(len(y) / sr),
            "pitch_summary": {
                "f0_min_hz": pitch["f0_min_hz"],
                "f0_max_hz": pitch["f0_max_hz"],
                "voiced_ratio": pitch["voiced_ratio"],
                "method": pitch.get("method", "unknown"),
            },
            "most_timbre_colored_f0": best,
            "formants_summary": {
                "dt_sec": form_cfg.dt_sec,
                "df_hz": form_cfg.df_hz,
                "mean_hz_over_time": form_mean,
                "examples_first3_windows": form_list[:3],
            },
        }

        if args.full:
            out["pitch_full"] = pitch
            out["formants_full"] = form
        return out

    report["A"] = analyze_one("A", a_path)
    report["I"] = analyze_one("I", i_path)
    report["other"] = analyze_one("other", o_path)

    text = json.dumps(report, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    if args.demo:
        (assets_dir / "demo_report.json").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
