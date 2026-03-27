#!/usr/bin/env python3

"""
Quickstart wake-word demo for sherpa-onnx with the wake word "小涂小涂".

Usage:
  uv sync
  uv run python python-api-examples/xiaotu-wakeup.py

Optional verification with a wave file:
  uv run python python-api-examples/xiaotu-wakeup.py --wave xiaotu.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first:")
    print("  uv sync")
    sys.exit(-1)


def add_windows_dll_directories() -> None:
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    exe_dir = Path(sys.executable).resolve().parent
    venv_root = exe_dir.parent
    candidates = [
        exe_dir,
        venv_root / "Lib" / "site-packages" / "sherpa_onnx" / "lib",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            os.add_dll_directory(str(candidate))


add_windows_dll_directories()

import sherpa_onnx


MODEL_NAME = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    f"{MODEL_NAME}.tar.bz2"
)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_model_paths(model_dir: Path) -> dict[str, Path]:
    return {
        "tokens": model_dir / "tokens.txt",
        "encoder": model_dir / "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
        "decoder": model_dir / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        "joiner": model_dir / "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    }


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive entry: {member.name}")
        tar.extractall(destination)


def download_with_progress(url: str, dest: Path) -> None:
    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percent = downloaded * 100 // total_size
        print(f"\rDownloading model... {percent:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def ensure_model(root: Path) -> Path:
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_dir = models_dir / MODEL_NAME
    if model_dir.is_dir():
        return model_dir

    archive = models_dir / f"{MODEL_NAME}.tar.bz2"
    if not archive.is_file():
        print(f"Model not found, downloading from {MODEL_URL}")
        download_with_progress(MODEL_URL, archive)

    print(f"Extracting {archive.name} ...")
    safe_extract_tar(archive, models_dir)

    if not model_dir.is_dir():
        raise RuntimeError(f"Expected extracted model directory: {model_dir}")

    return model_dir


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768
        return samples_float32, f.getframerate()


def create_keyword_spotter(args: argparse.Namespace, model_dir: Path) -> sherpa_onnx.KeywordSpotter:
    model_paths = get_model_paths(model_dir)
    for name, path in model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    if not args.keywords_file.is_file():
        raise FileNotFoundError(f"Missing keywords file: {args.keywords_file}")

    return sherpa_onnx.KeywordSpotter(
        tokens=str(model_paths["tokens"]),
        encoder=str(model_paths["encoder"]),
        decoder=str(model_paths["decoder"]),
        joiner=str(model_paths["joiner"]),
        num_threads=args.num_threads,
        max_active_paths=args.max_active_paths,
        keywords_file=str(args.keywords_file),
        keywords_score=args.keywords_score,
        keywords_threshold=args.keywords_threshold,
        num_trailing_blanks=args.num_trailing_blanks,
        provider=args.provider,
    )


def run_wave_file(
    keyword_spotter: sherpa_onnx.KeywordSpotter, wave_path: Path
) -> int:
    samples, sample_rate = read_wave(str(wave_path))
    tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)

    stream = keyword_spotter.create_stream()
    stream.accept_waveform(sample_rate, samples)
    stream.accept_waveform(sample_rate, tail_paddings)
    stream.input_finished()

    hits = 0
    while keyword_spotter.is_ready(stream):
        keyword_spotter.decode_stream(stream)
        result = keyword_spotter.get_result(stream)
        if result:
            hits += 1
            print(f"Detected: {result}")
            keyword_spotter.reset_stream(stream)

    if hits == 0:
        print("No keyword detected.")

    return hits


def list_devices() -> None:
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        print(f"[{index}] {device['name']}")


def run_microphone(keyword_spotter: sherpa_onnx.KeywordSpotter, device: int | None) -> None:
    devices = sd.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No microphone devices found.")

    input_device = device if device is not None else sd.default.device[0]
    if input_device is None or input_device < 0:
        raise RuntimeError("No default microphone device is configured.")

    print(f"Using microphone: [{input_device}] {devices[input_device]['name']}")
    print("Started! 请对着麦克风说: 小涂小涂")

    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)
    stream = keyword_spotter.create_stream()

    with sd.InputStream(
        channels=1,
        dtype="float32",
        samplerate=sample_rate,
        device=input_device,
    ) as audio_stream:
        hit_count = 0
        while True:
            samples, _ = audio_stream.read(samples_per_read)
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            while keyword_spotter.is_ready(stream):
                keyword_spotter.decode_stream(stream)
                result = keyword_spotter.get_result(stream)
                if result:
                    hit_count += 1
                    print(f"{hit_count}: Detected {result}")
                    keyword_spotter.reset_stream(stream)


def get_args() -> argparse.Namespace:
    root = get_repo_root()
    default_keywords = root / "keywords" / "xiaotu_xiaotu.txt"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--wave", type=Path, help="Optional wave file for verification")
    parser.add_argument(
        "--keywords-file",
        type=Path,
        default=default_keywords,
        help="Keyword file to use",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index. Omit to use the default microphone.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    parser.add_argument("--provider", type=str, default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--max-active-paths", type=int, default=4)
    parser.add_argument("--num-trailing-blanks", type=int, default=1)
    parser.add_argument("--keywords-score", type=float, default=1.0)
    parser.add_argument("--keywords-threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = get_args()

    if args.list_devices:
        list_devices()
        return

    root = get_repo_root()
    model_dir = ensure_model(root)
    keyword_spotter = create_keyword_spotter(args, model_dir)

    if args.wave is not None:
        if not args.wave.is_file():
            raise FileNotFoundError(f"Wave file not found: {args.wave}")
        run_wave_file(keyword_spotter, args.wave)
        return

    run_microphone(keyword_spotter, args.device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
