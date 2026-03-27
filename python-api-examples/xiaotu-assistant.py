#!/usr/bin/env python3

"""
Wake-word + VAD + Chinese ASR pipeline tuned for higher recall.

Pipeline:
  mono 16-bit PCM -> loudness normalization -> KWS/VAD -> transducer ASR
  -> modified_beam_search retry -> streaming CTC fallback

Usage:
  uv sync
  uv run python python-api-examples/xiaotu-assistant.py

Verification:
  uv run python python-api-examples/xiaotu-assistant.py --command-wave cmd.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import urllib.request
import wave
from pathlib import Path
from typing import Callable

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


SAMPLE_RATE = 16000
READ_SECONDS = 0.1
READ_SAMPLES = int(SAMPLE_RATE * READ_SECONDS)

KWS_MODEL_NAME = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
KWS_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    f"{KWS_MODEL_NAME}.tar.bz2"
)
KWS_MODEL_FILES = {
    "tokens": "tokens.txt",
    "encoder": "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "decoder": "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
    "joiner": "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
}

ASR_TRANSDUCER_NAME = "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"
ASR_TRANSDUCER_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{ASR_TRANSDUCER_NAME}.tar.bz2"
)
ASR_TRANSDUCER_FILES = {
    "tokens": "tokens.txt",
    "encoder": "encoder.int8.onnx",
    "decoder": "decoder.onnx",
    "joiner": "joiner.int8.onnx",
}

ASR_CTC_NAME = "sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30"
ASR_CTC_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{ASR_CTC_NAME}.tar.bz2"
)
ASR_CTC_FILES = {
    "tokens": "tokens.txt",
    "model": "model.int8.onnx",
}

VAD_MODEL_NAME = "silero_vad.onnx"
VAD_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{VAD_MODEL_NAME}"
)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:*") as tar:
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
        print(f"\rDownloading {dest.name} ... {percent:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def ensure_archive_dir(root: Path) -> Path:
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def ensure_tar_model(root: Path, model_name: str, url: str) -> Path:
    models_dir = ensure_archive_dir(root)
    model_dir = models_dir / model_name
    if model_dir.is_dir():
        return model_dir

    suffix = ".tar.bz2" if url.endswith(".tar.bz2") else ".tar.bz"
    archive = models_dir / f"{model_name}{suffix}"
    if not archive.is_file():
        download_with_progress(url, archive)

    print(f"Extracting {archive.name} ...")
    safe_extract_tar(archive, models_dir)
    if not model_dir.is_dir():
        raise RuntimeError(f"Expected extracted model directory: {model_dir}")

    return model_dir


def ensure_file(root: Path, name: str, url: str) -> Path:
    models_dir = ensure_archive_dir(root)
    path = models_dir / name
    if not path.is_file():
        download_with_progress(url, path)
    return path


def prepare_hotwords_file(
    root: Path, hotwords_file: Path | None, tokens_file: Path
) -> Path | None:
    if hotwords_file is None or not hotwords_file.is_file():
        return None

    valid_tokens = set()
    with open(tokens_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                valid_tokens.add(parts[0])

    kept_lines = []
    skipped_lines = []
    with open(hotwords_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            if all(token in valid_tokens for token in tokens):
                kept_lines.append(line)
            else:
                skipped_lines.append(line)

    if skipped_lines:
        print(f"跳过 {len(skipped_lines)} 条无法被当前模型编码的热词。")

    if not kept_lines:
        return None

    generated_dir = ensure_archive_dir(root) / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = generated_dir / f"{hotwords_file.stem}.{tokens_file.stem}.filtered.txt"
    with open(filtered_path, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
    return filtered_path


def read_wave(wave_filename: str) -> tuple[np.ndarray, int]:
    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768
        return samples_float32, f.getframerate()


def to_pcm16_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = samples.reshape(-1)
    if samples.dtype != np.int16:
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return samples


def normalize_pcm16(
    samples: np.ndarray,
    target_rms: float = 6000.0,
    max_gain: float = 8.0,
    peak_ceiling: float = 0.98,
) -> np.ndarray:
    x = to_pcm16_mono(samples).astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 64.0:
        return x / 32768.0

    rms = float(np.sqrt(np.mean(np.square(x)) + 1e-6))
    gain_from_rms = target_rms / max(rms, 1.0)
    gain_from_peak = (32767.0 * peak_ceiling) / max(peak, 1.0)
    gain = min(gain_from_rms, gain_from_peak, max_gain)
    normalized = np.clip(x * gain, -32768.0, 32767.0)
    return normalized.astype(np.float32) / 32768.0


def get_duration(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return float(len(samples)) / sample_rate


def create_kws(args: argparse.Namespace) -> sherpa_onnx.KeywordSpotter:
    root = get_repo_root()
    model_dir = ensure_tar_model(root, KWS_MODEL_NAME, KWS_MODEL_URL)
    return sherpa_onnx.KeywordSpotter(
        tokens=str(model_dir / KWS_MODEL_FILES["tokens"]),
        encoder=str(model_dir / KWS_MODEL_FILES["encoder"]),
        decoder=str(model_dir / KWS_MODEL_FILES["decoder"]),
        joiner=str(model_dir / KWS_MODEL_FILES["joiner"]),
        num_threads=args.num_threads,
        max_active_paths=args.kws_max_active_paths,
        keywords_file=str(args.keywords_file),
        keywords_score=args.kws_score,
        keywords_threshold=args.kws_threshold,
        num_trailing_blanks=args.kws_trailing_blanks,
        provider=args.provider,
    )


def create_transducer_recognizer(
    args: argparse.Namespace,
    decoding_method: str,
    hotwords_file: Path | None,
) -> sherpa_onnx.OnlineRecognizer:
    root = get_repo_root()
    model_dir = ensure_tar_model(root, ASR_TRANSDUCER_NAME, ASR_TRANSDUCER_URL)
    tokens_file = model_dir / ASR_TRANSDUCER_FILES["tokens"]
    filtered_hotwords = prepare_hotwords_file(root, hotwords_file, tokens_file)
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(tokens_file),
        encoder=str(model_dir / ASR_TRANSDUCER_FILES["encoder"]),
        decoder=str(model_dir / ASR_TRANSDUCER_FILES["decoder"]),
        joiner=str(model_dir / ASR_TRANSDUCER_FILES["joiner"]),
        num_threads=args.num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method=decoding_method,
        max_active_paths=args.asr_max_active_paths,
        hotwords_file=str(filtered_hotwords) if filtered_hotwords else "",
        hotwords_score=args.hotwords_score,
        blank_penalty=args.blank_penalty,
        provider=args.provider,
        enable_endpoint_detection=False,
        model_type="zipformer2",
        modeling_unit="cjkchar",
    )


def create_ctc_recognizer(args: argparse.Namespace) -> sherpa_onnx.OnlineRecognizer:
    root = get_repo_root()
    model_dir = ensure_tar_model(root, ASR_CTC_NAME, ASR_CTC_URL)
    return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=str(model_dir / ASR_CTC_FILES["tokens"]),
        model=str(model_dir / ASR_CTC_FILES["model"]),
        num_threads=args.num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        provider=args.provider,
        enable_endpoint_detection=False,
    )


def create_vad(args: argparse.Namespace) -> tuple[sherpa_onnx.VoiceActivityDetector, int]:
    root = get_repo_root()
    vad_model = ensure_file(root, VAD_MODEL_NAME, VAD_MODEL_URL)

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_model)
    config.silero_vad.threshold = args.vad_threshold
    config.silero_vad.min_silence_duration = args.vad_min_silence
    config.silero_vad.min_speech_duration = args.vad_min_speech
    config.silero_vad.max_speech_duration = args.vad_max_speech
    config.sample_rate = SAMPLE_RATE

    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
    return vad, config.silero_vad.window_size


def decode_online_recognizer(
    recognizer: sherpa_onnx.OnlineRecognizer,
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    stream = recognizer.create_stream()
    chunk_size = int(0.08 * sample_rate)
    for start in range(0, len(samples), chunk_size):
        stream.accept_waveform(sample_rate, samples[start : start + chunk_size])
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

    tail = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    return str(recognizer.get_result(stream)).strip()


def is_poor_result(text: str, duration: float) -> bool:
    stripped = "".join(text.split())
    if not stripped:
        return True
    if "<unk>" in stripped.lower():
        return True
    return duration >= 1.6 and len(stripped) <= 1


class AsrPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.hotwords_file = (
            args.hotwords_file
            if args.hotwords_file.is_file() and args.hotwords_file.stat().st_size > 0
            else None
        )
        self._greedy: sherpa_onnx.OnlineRecognizer | None = None
        self._beam: sherpa_onnx.OnlineRecognizer | None = None
        self._ctc: sherpa_onnx.OnlineRecognizer | None = None

    def greedy(self) -> sherpa_onnx.OnlineRecognizer:
        if self._greedy is None:
            self._greedy = create_transducer_recognizer(
                self.args, "greedy_search", None
            )
        return self._greedy

    def beam(self) -> sherpa_onnx.OnlineRecognizer:
        if self._beam is None:
            self._beam = create_transducer_recognizer(
                self.args, "modified_beam_search", self.hotwords_file
            )
        return self._beam

    def ctc(self) -> sherpa_onnx.OnlineRecognizer:
        if self._ctc is None:
            self._ctc = create_ctc_recognizer(self.args)
        return self._ctc

    def decode(self, samples: np.ndarray) -> tuple[str, str]:
        duration = get_duration(samples)
        if self.hotwords_file is not None:
            text = decode_online_recognizer(self.beam(), samples)
            if text and not is_poor_result(text, duration):
                return text, "transducer+hotwords"
        else:
            text = decode_online_recognizer(self.greedy(), samples)
            if text and not is_poor_result(text, duration):
                return text, "transducer-greedy"

        text = decode_online_recognizer(self.beam(), samples)
        if text and not is_poor_result(text, duration):
            return text, "transducer-modified_beam_search"

        boosted = normalize_pcm16(
            np.clip(samples * 32768.0, -32768.0, 32767.0).astype(np.int16),
            target_rms=8000.0,
            max_gain=10.0,
        )
        text = decode_online_recognizer(self.ctc(), boosted)
        if text:
            return text, "streaming-ctc-fallback"

        return "", "empty"


def list_devices() -> None:
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        print(f"[{index}] {device['name']}")


def capture_command(
    audio_stream: sd.InputStream,
    args: argparse.Namespace,
    preprocess: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray | None:
    vad, window_size = create_vad(args)
    pending = np.empty(0, dtype=np.float32)
    deadline = time.monotonic() + args.max_command_wait

    while time.monotonic() < deadline:
        samples, _ = audio_stream.read(READ_SAMPLES)
        samples = preprocess(samples)
        pending = np.concatenate([pending, samples])

        while len(pending) >= window_size:
            vad.accept_waveform(pending[:window_size])
            pending = pending[window_size:]

        while not vad.empty():
            segment = np.copy(vad.front.samples)
            vad.pop()
            if get_duration(segment) < args.min_command_duration:
                print(
                    f"忽略过短句段: {get_duration(segment):.2f}s < "
                    f"{args.min_command_duration:.2f}s"
                )
                continue
            return segment

    if len(pending) > 0:
        padded = np.zeros(window_size, dtype=np.float32)
        padded[: len(pending)] = pending
        vad.accept_waveform(padded)

    vad.flush()
    while not vad.empty():
        segment = np.copy(vad.front.samples)
        vad.pop()
        if get_duration(segment) >= args.min_command_duration:
            return segment

    return None


def make_preprocessor(args: argparse.Namespace) -> Callable[[np.ndarray], np.ndarray]:
    def preprocess(samples: np.ndarray) -> np.ndarray:
        pcm16 = to_pcm16_mono(samples)
        if args.disable_normalization:
            return pcm16.astype(np.float32) / 32768.0
        return normalize_pcm16(
            pcm16,
            target_rms=args.normalization_target_rms,
            max_gain=args.normalization_max_gain,
        )

    return preprocess


def get_args() -> argparse.Namespace:
    root = get_repo_root()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--keywords-file",
        type=Path,
        default=root / "keywords" / "xiaotu_xiaotu.txt",
    )
    parser.add_argument(
        "--hotwords-file",
        type=Path,
        default=root / "hotwords" / "common_zh.txt",
    )
    parser.add_argument("--command-wave", type=Path, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--provider", type=str, default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--kws-score", type=float, default=1.5)
    parser.add_argument("--kws-threshold", type=float, default=0.30)
    parser.add_argument("--kws-max-active-paths", type=int, default=8)
    parser.add_argument("--kws-trailing-blanks", type=int, default=1)
    parser.add_argument("--hotwords-score", type=float, default=1.8)
    parser.add_argument("--asr-max-active-paths", type=int, default=8)
    parser.add_argument("--blank-penalty", type=float, default=0.15)
    parser.add_argument("--vad-threshold", type=float, default=0.25)
    parser.add_argument("--vad-min-silence", type=float, default=0.60)
    parser.add_argument("--vad-min-speech", type=float, default=0.45)
    parser.add_argument("--vad-max-speech", type=float, default=12.0)
    parser.add_argument("--max-command-wait", type=float, default=8.0)
    parser.add_argument("--min-command-duration", type=float, default=0.80)
    parser.add_argument("--normalization-target-rms", type=float, default=6000.0)
    parser.add_argument("--normalization-max-gain", type=float, default=8.0)
    parser.add_argument("--disable-normalization", action="store_true")
    return parser.parse_args()


def run_command_wave(args: argparse.Namespace, asr: AsrPipeline) -> None:
    if not args.command_wave or not args.command_wave.is_file():
        raise FileNotFoundError(f"Wave file not found: {args.command_wave}")

    samples, sample_rate = read_wave(str(args.command_wave))
    if sample_rate != SAMPLE_RATE:
        print(f"Resample inside sherpa-onnx: {sample_rate} -> {SAMPLE_RATE}")

    command = samples if args.disable_normalization else normalize_pcm16(
        np.clip(samples * 32768.0, -32768.0, 32767.0).astype(np.int16),
        target_rms=args.normalization_target_rms,
        max_gain=args.normalization_max_gain,
    )
    text, strategy = asr.decode(command)
    print(f"strategy: {strategy}")
    print(f"text: {text}")


def run_microphone(args: argparse.Namespace) -> None:
    devices = sd.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No microphone devices found.")

    input_device = args.device if args.device is not None else sd.default.device[0]
    if input_device is None or input_device < 0:
        raise RuntimeError("No default microphone device is configured.")

    preprocess = make_preprocessor(args)
    kws = create_kws(args)
    kws_stream = kws.create_stream()
    asr = AsrPipeline(args)

    print(f"Using microphone: [{input_device}] {devices[input_device]['name']}")
    print("已启用: 单声道 16-bit PCM + 归一化 + VAD + hotwords + fallback")
    print("先说唤醒词: 小涂小涂")

    with sd.InputStream(
        channels=1,
        dtype="int16",
        samplerate=SAMPLE_RATE,
        device=input_device,
    ) as audio_stream:
        while True:
            samples, _ = audio_stream.read(READ_SAMPLES)
            samples = preprocess(samples)
            kws_stream.accept_waveform(SAMPLE_RATE, samples)

            detected = ""
            while kws.is_ready(kws_stream):
                kws.decode_stream(kws_stream)
                detected = kws.get_result(kws_stream)
                if detected:
                    break

            if not detected:
                continue

            print(f"唤醒成功: {detected}")
            kws.reset_stream(kws_stream)
            print("请说指令 ...")

            command = capture_command(audio_stream, args, preprocess)
            if command is None:
                print("没有抓到足够长的语音指令，继续待命。")
                continue

            text, strategy = asr.decode(command)
            if text:
                print(f"[{strategy}] {text}")
            else:
                print("本轮识别为空，已用重试与 fallback 兜底，但仍未得到结果。")


def main() -> None:
    args = get_args()
    if args.list_devices:
        list_devices()
        return

    if not args.keywords_file.is_file():
        raise FileNotFoundError(f"Missing keywords file: {args.keywords_file}")

    asr = AsrPipeline(args)
    if args.command_wave is not None:
        run_command_wave(args, asr)
        return

    run_microphone(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
