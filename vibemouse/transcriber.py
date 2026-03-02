from __future__ import annotations

import importlib
import os
import re
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, cast

from vibemouse.config import AppConfig


class SenseVoiceTranscriber:
    def __init__(self, config: AppConfig) -> None:
        self._config: AppConfig = config
        self._transcriber: _TranscriberProtocol | None = None
        self._transcriber_lock: Lock = Lock()
        self.device_in_use: str = config.device
        self.backend_in_use: str = "unknown"

    def transcribe(self, audio_path: Path) -> str:
        self._ensure_transcriber_loaded()
        if self._transcriber is None:
            raise RuntimeError("SenseVoice transcriber is not initialized")
        return self._transcriber.transcribe(audio_path)

    def prewarm(self) -> None:
        self._ensure_transcriber_loaded()

    def _ensure_transcriber_loaded(self) -> None:
        if self._transcriber is not None:
            return

        with self._transcriber_lock:
            if self._transcriber is not None:
                return

            backend = self._config.transcriber_backend
            if backend == "auto":
                self._build_auto_backend()
                return

            if backend == "funasr_onnx":
                self._build_funasr_onnx_backend()
                return

            raise RuntimeError(
                f"Unsupported backend {backend!r}. Use auto or funasr_onnx."
            )

    def _build_auto_backend(self) -> None:
        try:
            self._build_funasr_onnx_backend()
        except Exception as error:
            raise RuntimeError(f"Failed to initialize ONNX backend: {error}") from error

    def _build_funasr_onnx_backend(self) -> None:
        backend = _FunASRONNXBackend(self._config)
        self._transcriber = backend
        self.device_in_use = backend.device_in_use
        self.backend_in_use = "funasr_onnx"

class _FunASRONNXBackend:
    def __init__(self, config: AppConfig) -> None:
        self._config: AppConfig = config
        self._model: _ONNXSenseVoiceModel | None = None
        self._postprocess: _PostprocessFn | None = None
        self._load_lock: Lock = Lock()
        self.device_in_use: str = "cpu"
        self._ensure_model_loaded()

    def transcribe(self, audio_path: Path) -> str:
        if self._model is None:
            raise RuntimeError("funasr_onnx SenseVoice model is not initialized")
        if self._postprocess is None:
            raise RuntimeError("funasr postprocess function is not initialized")

        chunk_seconds = self._read_chunk_seconds()
        if chunk_seconds > 0:
            long_text = self._transcribe_by_chunks(audio_path, chunk_seconds)
            if long_text is not None:
                return long_text

        textnorm = "withitn" if self._config.use_itn else "woitn"
        result = self._model(
            str(audio_path),
            language=self._config.language,
            textnorm=textnorm,
        )
        if not result:
            return ""

        raw_text = result[0]
        return self._normalize_transcript(self._postprocess(raw_text)).strip()

    def _read_chunk_seconds(self) -> int:
        raw = os.getenv("VIBEMOUSE_ONNX_CHUNK_S", "14").strip()
        try:
            value = int(raw)
        except ValueError:
            return 14
        if value <= 0:
            return 0
        return min(value, 120)

    def _read_chunk_overlap_seconds(self) -> float:
        raw = os.getenv("VIBEMOUSE_ONNX_CHUNK_OVERLAP_S", "2.0").strip()
        try:
            value = float(raw)
        except ValueError:
            return 2.0
        if value < 0:
            return 0.0
        return min(value, 8.0)

    def _transcribe_by_chunks(self, audio_path: Path, chunk_seconds: int) -> str | None:
        if self._model is None or self._postprocess is None:
            return None

        try:
            soundfile_module = importlib.import_module("soundfile")
            read_fn = cast(_SoundFileReadFn, getattr(soundfile_module, "read"))
            write_fn = cast(_SoundFileWriteFn, getattr(soundfile_module, "write"))
            audio_obj, sample_rate = read_fn(str(audio_path), dtype="float32")
            audio = cast(Any, audio_obj)
        except Exception:
            return None

        total_samples = len(audio)
        if total_samples <= 0:
            return ""

        chunk_samples = chunk_seconds * sample_rate
        if chunk_samples <= 0 or total_samples <= chunk_samples:
            return None

        overlap_samples = int(self._read_chunk_overlap_seconds() * sample_rate)
        stride_samples = max(1, chunk_samples - overlap_samples)

        textnorm = "withitn" if self._config.use_itn else "woitn"
        merged_text = ""
        with tempfile.TemporaryDirectory(prefix="vibemouse-onnx-") as tmp:
            tmp_dir = Path(tmp)
            chunk_index = 0
            start = 0
            while start < total_samples:
                end = min(start + chunk_samples, total_samples)
                segment = audio[start:end]
                if len(segment) < max(1, sample_rate // 6):
                    break
                chunk_path = tmp_dir / f"chunk_{chunk_index:03d}.wav"
                chunk_index += 1
                write_fn(str(chunk_path), segment, sample_rate)
                result = self._model(
                    str(chunk_path),
                    language=self._config.language,
                    textnorm=textnorm,
                )
                if not result:
                    if end >= total_samples:
                        break
                    start += stride_samples
                    continue
                part = self._postprocess(result[0]).strip()
                part = self._normalize_transcript(part).strip()
                if part:
                    merged_text = self._merge_chunk_text(merged_text, part)
                if end >= total_samples:
                    break
                start += stride_samples
        return merged_text.strip()

    @staticmethod
    def _merge_chunk_text(existing: str, incoming: str) -> str:
        if not existing:
            return incoming
        if not incoming:
            return existing
        if incoming in existing:
            return existing

        max_overlap = min(len(existing), len(incoming), 60)
        overlap = 0
        for size in range(max_overlap, 0, -1):
            if existing.endswith(incoming[:size]):
                overlap = size
                break

        if overlap > 0:
            return existing + incoming[overlap:]

        return existing + " " + incoming

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return
            try:
                SenseVoiceSmall = self._load_onnx_class()
                postprocess = self._load_postprocess()
            except Exception as error:
                raise RuntimeError(
                    "funasr_onnx backend is not available in current environment"
                ) from error

            requested_path = self._resolve_onnx_model_dir()
            self._ensure_tokenizer_file(requested_path)
            device_id = self._resolve_onnx_device_id(self._config.device)

            try:
                model = SenseVoiceSmall(
                    model_dir=str(requested_path),
                    batch_size=1,
                    device_id=device_id,
                    quantize=True,
                    cache_dir=None,
                )
                self._model = model
                self._postprocess = postprocess
                self.device_in_use = self._resolve_device_label(self._config.device)
                return
            except Exception as primary_error:
                if not self._config.fallback_to_cpu:
                    raise RuntimeError(
                        f"Failed to load funasr_onnx backend on {self._config.device}: {primary_error}"
                    ) from primary_error

            try:
                model = SenseVoiceSmall(
                    model_dir=str(requested_path),
                    batch_size=1,
                    device_id="-1",
                    quantize=True,
                    cache_dir=None,
                )
            except Exception as cpu_error:
                raise RuntimeError(
                    f"Failed to load funasr_onnx backend on {self._config.device} and cpu fallback: {cpu_error}"
                ) from cpu_error

            self._model = model
            self._postprocess = postprocess
            self.device_in_use = "cpu"

    def _resolve_onnx_model_dir(self) -> Path:
        raw_model = self._config.model_name
        canonical_model = raw_model
        if raw_model == "iic/SenseVoiceSmall":
            canonical_model = "iic/SenseVoiceSmall-onnx"

        if canonical_model.startswith("iic/"):
            local_cache = (
                Path.home()
                / ".cache"
                / "modelscope"
                / "hub"
                / "models"
                / canonical_model.replace("/", os.sep)
            )
            if not local_cache.exists():
                raise RuntimeError(
                    f"ONNX model not found locally: {local_cache}. "
                    + "Please pre-download model iic/SenseVoiceSmall-onnx."
                )
            if not self._contains_onnx_model(local_cache):
                raise RuntimeError(
                    f"Local ONNX model directory {local_cache} is missing model_quant.onnx/model.onnx"
                )
            return local_cache

        path_candidate = Path(canonical_model)
        if not path_candidate.exists():
            return path_candidate

        if self._contains_onnx_model(path_candidate):
            return path_candidate

        raise RuntimeError(
            f"ONNX model directory {path_candidate} exists but model_quant.onnx/model.onnx is missing"
        )

    @staticmethod
    def _contains_onnx_model(model_dir: Path) -> bool:
        return (model_dir / "model_quant.onnx").exists() or (
            model_dir / "model.onnx"
        ).exists()

    @staticmethod
    def _resolve_onnx_device_id(device: str) -> str:
        normalized = device.strip().lower()
        if normalized == "cpu":
            return "-1"
        if normalized.startswith("cuda"):
            parts = normalized.split(":", 1)
            return parts[1] if len(parts) > 1 and parts[1] else "0"
        return "-1"

    @staticmethod
    def _resolve_device_label(device: str) -> str:
        normalized = device.strip().lower()
        if normalized.startswith("cuda"):
            return normalized
        return "cpu"

    def _ensure_tokenizer_file(self, model_dir: Path) -> None:
        target = model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        if target.exists():
            return

        fallback = (
            Path.home()
            / ".cache/modelscope/hub/models/iic/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model"
        )
        if fallback.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            _ = target.write_bytes(fallback.read_bytes())
            return

        raise RuntimeError(
            "Tokenizer file chn_jpn_yue_eng_ko_spectok.bpe.model is missing and no fallback was found"
        )

    @staticmethod
    def _load_onnx_class() -> _ONNXSenseVoiceCtor:
        module = importlib.import_module("funasr_onnx")
        return cast(_ONNXSenseVoiceCtor, getattr(module, "SenseVoiceSmall"))

    @staticmethod
    def _load_postprocess() -> _PostprocessFn:
        try:
            post_module = importlib.import_module("funasr.utils.postprocess_utils")
            return cast(
                _PostprocessFn,
                getattr(post_module, "rich_transcription_postprocess"),
            )
        except Exception:
            return cast(_PostprocessFn, lambda text: text)

    @staticmethod
    def _normalize_transcript(text: str) -> str:
        cleaned = text
        cleaned = re.sub(r"<\|[^|]*\|>", " ", cleaned)
        cleaned = cleaned.replace("<||>", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()


class _TranscriberProtocol(Protocol):
    device_in_use: str

    def transcribe(self, audio_path: Path) -> str: ...


class _PostprocessFn(Protocol):
    def __call__(self, text: str) -> str: ...


class _ONNXSenseVoiceModel(Protocol):
    def __call__(
        self,
        wav_content: str,
        *,
        language: str,
        textnorm: str,
    ) -> list[str]: ...


class _ONNXSenseVoiceCtor(Protocol):
    def __call__(
        self,
        *,
        model_dir: str,
        batch_size: int,
        device_id: str,
        quantize: bool,
        cache_dir: str | None,
    ) -> _ONNXSenseVoiceModel: ...


class _SoundFileReadFn(Protocol):
    def __call__(self, file: str, *, dtype: str = "float32") -> tuple[Any, int]: ...


class _SoundFileWriteFn(Protocol):
    def __call__(self, file: str, data: object, samplerate: int) -> None: ...
