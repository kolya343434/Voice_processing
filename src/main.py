from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_io import load_audio_mono, save_wav
from plotting import save_spectrogram
from v2_demo import generate_inventory_demo
from v2_synth import Variant2Config, load_phone_segments, normalize_audio, synth_concat, synth_crossfade
from voice_analysis import (
    FormantsConfig,
    HarmonicsConfig,
    PitchConfig,
    estimate_pitch_cepstrum,
    formants_over_time,
    most_timbre_colored_f0,
)
from voice_demo import generate_demo_variant1


def _read_cfg(repo_root: Path, variant: int) -> dict:
    cfg_path = repo_root / "config" / f"variant{variant}.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Lab 10: voice processing (variant 1 or 2).")
    parser.add_argument("--variant", type=int, default=1, choices=[1, 2], help="Lab variant (1 or 2).")
    parser.add_argument("--demo", action="store_true", help="Generate demo inputs and run.")
    parser.add_argument("--json-out", type=str, default=None, help="Write report JSON to file.")

    # Variant 1
    parser.add_argument("--A", dest="a_path", type=str, default=None, help="Variant 1: path to vowel A wav.")
    parser.add_argument("--I", dest="i_path", type=str, default=None, help="Variant 1: path to vowel I wav.")
    parser.add_argument("--other", dest="o_path", type=str, default=None, help="Variant 1: path to bark/meow/etc wav.")
    parser.add_argument("--full", action="store_true", help="Variant 1: include full arrays in JSON.")

    # Variant 2
    parser.add_argument("--samples-dir", type=str, default=None, help="Variant 2: directory with <phone>.wav files.")
    parser.add_argument("--mode", type=str, default="both", choices=["concat", "crossfade", "both"], help="Variant 2: synthesis mode.")
    parser.add_argument("--crossfade-ms", type=int, default=None, help="Variant 2: crossfade in ms (overrides config).")
    parser.add_argument("--check-inventory", action="store_true", help="Variant 2: only check that all inventory WAV files exist.")
    args = parser.parse_args()

    variant = int(args.variant)
    cfg = _read_cfg(repo_root, variant)
    if int(cfg.get("variant", variant)) != variant:
        raise SystemExit(f"Config mismatch for variant {variant}.")

    sr = int(cfg["sample_rate"])
    assets_dir = repo_root / "assets"
    outputs_dir = repo_root / "outputs"

    if variant == 2:
        inv = cfg["inventory"]
        samples_dir = Path(args.samples_dir) if args.samples_dir else (repo_root / inv["dir"])
        phones_inventory = list(inv["phones"])
        phrase_phones = list(cfg["phrase"]["phones"])
        crossfade_ms = int(args.crossfade_ms) if args.crossfade_ms is not None else int(cfg.get("crossfade_ms", 25))

        if args.demo:
            generate_inventory_demo(samples_dir, phones_inventory, sr=sr)

        missing = [ph for ph in phones_inventory if not (samples_dir / f"{ph}.wav").exists()]
        if args.check_inventory:
            text = json.dumps(
                {
                    "variant": 2,
                    "samples_dir": str(samples_dir),
                    "inventory_count": len(phones_inventory),
                    "missing_count": len(missing),
                    "missing": missing,
                },
                ensure_ascii=False,
                indent=2,
            )
            print(text)
            return 0
        if missing:
            raise SystemExit(
                f"Missing {len(missing)} inventory files in {samples_dir}. "
                f"Run --demo to generate placeholders or record your WAVs. "
                f"Example missing: {', '.join(missing[:10])}"
            )

        v2cfg = Variant2Config(sample_rate=sr, crossfade_ms=crossfade_ms)
        segments = load_phone_segments(phrase_phones, samples_dir, v2cfg)
        crossfade_samples = int(round(sr * (crossfade_ms / 1000.0)))
        stft_cfg = cfg["stft"]

        report: dict[str, object] = {
            "lab": 10,
            "variant": 2,
            "phrase_text": str(cfg["phrase"]["text"]),
            "phones_inventory_count": int(len(phones_inventory)),
            "phrase_phones": phrase_phones,
            "sample_rate": sr,
            "crossfade_ms": crossfade_ms,
            "paths": {"samples_dir": str(samples_dir)},
            "outputs": {},
        }

        if args.mode in ("concat", "both"):
            y = normalize_audio(synth_concat(segments))
            out_wav = outputs_dir / "v2_synth_concat.wav"
            save_wav(out_wav, y, sr)
            if args.demo:
                save_wav(assets_dir / "v2_synth_concat.wav", y, sr)
            save_spectrogram(
                assets_dir / "v2_spectrogram_concat.png",
                y,
                sr,
                n_fft=int(stft_cfg["n_fft"]),
                hop_length=int(stft_cfg["hop_length"]),
                title="Spectrogram (variant2 concat, Hann, log-freq)",
                log_freq=True,
            )
            report["outputs"] = {
                **report["outputs"],
                "concat_wav": str(out_wav),
                "concat_spectrogram_png": str(assets_dir / "v2_spectrogram_concat.png"),
            }

        if args.mode in ("crossfade", "both"):
            y = normalize_audio(synth_crossfade(segments, crossfade_samples=crossfade_samples))
            out_wav = outputs_dir / "v2_synth_crossfade.wav"
            save_wav(out_wav, y, sr)
            if args.demo:
                save_wav(assets_dir / "v2_synth_crossfade.wav", y, sr)
            save_spectrogram(
                assets_dir / "v2_spectrogram_crossfade.png",
                y,
                sr,
                n_fft=int(stft_cfg["n_fft"]),
                hop_length=int(stft_cfg["hop_length"]),
                title="Spectrogram (variant2 crossfade, Hann, log-freq)",
                log_freq=True,
            )
            report["outputs"] = {
                **report["outputs"],
                "crossfade_wav": str(out_wav),
                "crossfade_spectrogram_png": str(assets_dir / "v2_spectrogram_crossfade.png"),
            }

        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        if args.json_out:
            Path(args.json_out).write_text(text, encoding="utf-8")
        if args.demo:
            (assets_dir / "v2_demo_report.json").write_text(text, encoding="utf-8")
        return 0

    # Variant 1
    spec_cfg = cfg["spectrogram"]
    pitch_cfg = PitchConfig(**cfg["pitch"])
    harm_cfg = HarmonicsConfig(**cfg["harmonics"])
    form_cfg = FormantsConfig(**cfg["formants"])

    samples_dir = repo_root / "samples"

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
