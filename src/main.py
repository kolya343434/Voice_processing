from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_io import save_wav
from demo import generate_demo_samples
from plotting import save_spectrogram
from synth import SynthConfig, load_phone_segments, normalize_audio, synth_concat, synth_crossfade


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "config" / "variant2.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(description="Lab 10 (variant 2): Speech synthesizer from phoneme samples.")
    parser.add_argument("--samples-dir", type=str, default=str(repo_root / "samples"), help="Directory with <phone>.wav files.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo samples and synthesize the phrase.")
    parser.add_argument("--mode", type=str, default="both", choices=["concat", "crossfade", "both"], help="Synthesis mode.")
    parser.add_argument("--crossfade-ms", type=int, default=int(cfg.get("crossfade_ms", 20)), help="Crossfade duration in ms.")
    parser.add_argument("--json-out", type=str, default=None, help="Write report JSON to a file.")
    args = parser.parse_args()

    if int(cfg.get("variant", 2)) != 2:
        raise SystemExit("This repo is prepared for lab 10, variant 2.")

    sample_rate = int(cfg.get("sample_rate", 22050))
    phones = list(cfg["phrase"]["phones"])
    stft_cfg = cfg["stft"]

    synth_cfg = SynthConfig(sample_rate=sample_rate, crossfade_ms=int(args.crossfade_ms))
    crossfade_samples = int(round(sample_rate * (synth_cfg.crossfade_ms / 1000.0)))

    if args.demo:
        generate_demo_samples(args.samples_dir, phones, sr=sample_rate)

    segments = load_phone_segments(phones, args.samples_dir, synth_cfg)

    outputs_dir = repo_root / "outputs"
    assets_dir = repo_root / "assets"

    report: dict[str, object] = {
        "lab": 10,
        "variant": 2,
        "phrase_text": str(cfg["phrase"]["text"]),
        "phones": phones,
        "sample_rate": sample_rate,
        "crossfade_ms": synth_cfg.crossfade_ms,
        "outputs": {},
    }

    if args.mode in ("concat", "both"):
        y = normalize_audio(synth_concat(segments))
        out_wav = outputs_dir / "synth_concat.wav"
        save_wav(out_wav, y, sample_rate)
        save_spectrogram(
            assets_dir / "spectrogram_concat.png",
            y,
            sample_rate,
            n_fft=int(stft_cfg["n_fft"]),
            hop_length=int(stft_cfg["hop_length"]),
            title="Spectrogram (concat, Hann, log-freq)",
            log_freq=True,
        )
        report["outputs"] = {**report["outputs"], "synth_concat_wav": str(out_wav), "spectrogram_concat_png": str(assets_dir / "spectrogram_concat.png")}

    if args.mode in ("crossfade", "both"):
        y = normalize_audio(synth_crossfade(segments, crossfade_samples=crossfade_samples))
        out_wav = outputs_dir / "synth_crossfade.wav"
        save_wav(out_wav, y, sample_rate)
        save_spectrogram(
            assets_dir / "spectrogram_crossfade.png",
            y,
            sample_rate,
            n_fft=int(stft_cfg["n_fft"]),
            hop_length=int(stft_cfg["hop_length"]),
            title="Spectrogram (crossfade, Hann, log-freq)",
            log_freq=True,
        )
        report["outputs"] = {**report["outputs"], "synth_crossfade_wav": str(out_wav), "spectrogram_crossfade_png": str(assets_dir / "spectrogram_crossfade.png")}

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    if args.demo:
        (assets_dir / "demo_report.json").write_text(text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
