from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Iterable, Mapping
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray


AudioFrame = NDArray[np.float32]


@dataclass
class AudioRecording:
    path: Path
    duration_s: float


class _AudioStream(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class _SoundDeviceModule(Protocol):
    def InputStream(self, **kwargs: object) -> _AudioStream: ...

    def query_devices(self) -> object: ...


class _SoundFileModule(Protocol):
    def write(self, file: str | Path, data: AudioFrame, samplerate: int) -> None: ...


class AudioRecorder:
    def __init__(
        self, sample_rate: int, channels: int, dtype: str, temp_dir: Path
    ) -> None:
        self._sample_rate: int = sample_rate
        self._channels: int = channels
        self._dtype: str = dtype
        self._temp_dir: Path = temp_dir
        self._sd: _SoundDeviceModule | None = None
        self._sf: _SoundFileModule | None = None
        self._lock: threading.Lock = threading.Lock()
        self._frames: list[AudioFrame] = []
        self._stream: _AudioStream | None = None
        self._recording: bool = False
        self._active_sample_rate: int = sample_rate
        self._preferred_device_raw: str = os.getenv("VIBEMOUSE_AUDIO_DEVICE", "").strip()

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def start(self) -> None:
        self._ensure_audio_modules()
        with self._lock:
            if self._recording:
                return
            try:
                self._temp_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise RuntimeError(
                    f"Failed to create temp audio directory {self._temp_dir}: {error}"
                ) from error
            self._frames = []
            if self._sd is None:
                raise RuntimeError("Audio input module not initialized")
            stream, active_sample_rate = self._open_input_stream_with_fallback()
            stream.start()
            self._stream = stream
            self._recording = True
            self._active_sample_rate = active_sample_rate

    def stop_and_save(self) -> AudioRecording | None:
        with self._lock:
            if not self._recording:
                return None
            stream = self._stream
            self._stream = None
            self._recording = False

        if stream is not None:
            stream.stop()
            stream.close()

        with self._lock:
            if not self._frames:
                return None
            audio = np.concatenate(self._frames, axis=0)
            self._frames = []

        out_path = self._temp_dir / f"recording_{uuid4().hex}.wav"
        if self._sf is None:
            raise RuntimeError("Audio write module not initialized")
        try:
            self._sf.write(out_path, audio, self._active_sample_rate)
        except Exception as error:
            raise RuntimeError(
                f"Failed to write recording to {out_path}: {error}"
            ) from error
        duration = float(len(audio) / self._active_sample_rate)
        return AudioRecording(path=out_path, duration_s=duration)

    def _open_input_stream_with_fallback(self) -> tuple[_AudioStream, int]:
        if self._sd is None:
            raise RuntimeError("Audio input module not initialized")

        base_kwargs: dict[str, object] = {
            "dtype": self._dtype,
            "callback": self._callback,
        }

        attempts: list[str] = []
        last_error: Exception | None = None
        for candidate in self._build_stream_candidates():
            kwargs = {
                **base_kwargs,
                "samplerate": candidate["sample_rate"],
                "channels": candidate["channels"],
            }
            device = candidate.get("device")
            if isinstance(device, int):
                kwargs["device"] = device

            try:
                stream = self._sd.InputStream(**kwargs)
                print(
                    "Audio input opened: "
                    + f"device={candidate['label']}, sample_rate={candidate['sample_rate']}, channels={candidate['channels']}"
                )
                return stream, int(candidate["sample_rate"])
            except Exception as error:
                last_error = error
                attempts.append(
                    f"{candidate['label']}@{candidate['sample_rate']}Hz/{candidate['channels']}ch: {error}"
                )

        hint = ""
        if self._preferred_device_raw:
            hint = (
                " Check VIBEMOUSE_AUDIO_DEVICE; current value="
                + repr(self._preferred_device_raw)
                + "."
            )
        detail = " | ".join(attempts[-6:]) if attempts else "no attempts"
        raise RuntimeError(
            "Failed to open any microphone input stream. " + detail + hint
        ) from last_error

    def _build_stream_candidates(self) -> list[dict[str, object]]:
        default_rates = [self._sample_rate, 16000, 48000, 44100]
        candidates: list[dict[str, object]] = []

        for rate in self._unique_positive_ints(default_rates):
            candidates.append(
                {
                    "label": "default",
                    "sample_rate": rate,
                    "channels": self._channels,
                }
            )

        fallback_configs = self._pick_fallback_input_devices()
        preferred_index = self._resolve_preferred_device_index(fallback_configs)
        ordered = fallback_configs
        if preferred_index is not None:
            ordered = sorted(
                fallback_configs,
                key=lambda item: 0 if item["index"] == preferred_index else 1,
            )

        for item in ordered:
            sample_rates = self._unique_positive_ints(
                [
                    self._sample_rate,
                    int(item["sample_rate"]),
                    16000,
                    48000,
                    44100,
                ]
            )
            for rate in sample_rates:
                candidates.append(
                    {
                        "label": f"{item['name']}#{item['index']}",
                        "device": int(item["index"]),
                        "sample_rate": rate,
                        "channels": int(item["channels"]),
                    }
                )

        deduped: list[dict[str, object]] = []
        seen: set[tuple[object, int, int]] = set()
        for item in candidates:
            key = (
                item.get("device"),
                int(item["sample_rate"]),
                int(item["channels"]),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _pick_fallback_input_devices(self) -> list[dict[str, object]]:
        if self._sd is None:
            return []

        query_devices = getattr(self._sd, "query_devices", None)
        if not callable(query_devices):
            return []

        try:
            devices_obj = query_devices()
        except Exception:
            return []

        if not isinstance(devices_obj, Iterable):
            return []

        devices: list[dict[str, object]] = []

        for index, item in enumerate(devices_obj):
            if not isinstance(item, Mapping):
                continue

            max_input = self._to_int(item.get("max_input_channels"), default=0)
            if max_input <= 0:
                continue

            sample_rate = self._to_int(
                item.get("default_samplerate"),
                default=self._sample_rate,
            )
            channels = max(1, min(self._channels, max_input))
            name_raw = item.get("name")
            name = str(name_raw).strip() if name_raw is not None else f"device-{index}"
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "sample_rate": sample_rate,
                    "channels": channels,
                }
            )

        return devices

    def _resolve_preferred_device_index(
        self, devices: list[dict[str, object]]
    ) -> int | None:
        raw = self._preferred_device_raw
        if not raw:
            return None

        try:
            numeric = int(raw)
        except ValueError:
            numeric = None

        if isinstance(numeric, int):
            for item in devices:
                if int(item["index"]) == numeric:
                    return numeric

        lowered = raw.lower()
        for item in devices:
            name = str(item.get("name", "")).lower()
            if lowered and lowered in name:
                return int(item["index"])

        print(
            "Preferred microphone not found: "
            + f"VIBEMOUSE_AUDIO_DEVICE={raw!r}. Falling back to auto selection."
        )
        return None

    @staticmethod
    def _unique_positive_ints(values: list[int]) -> list[int]:
        output: list[int] = []
        seen: set[int] = set()
        for item in values:
            if item <= 0:
                continue
            if item in seen:
                continue
            seen.add(item)
            output.append(item)
        return output

    @staticmethod
    def _to_int(value: object, *, default: int) -> int:
        if isinstance(value, int):
            return value if value > 0 else default
        if isinstance(value, float):
            parsed = int(value)
            return parsed if parsed > 0 else default
        if isinstance(value, str):
            try:
                parsed = int(float(value.strip()))
            except ValueError:
                return default
            return parsed if parsed > 0 else default
        return default

    def cancel(self) -> None:
        with self._lock:
            if not self._recording:
                self._frames = []
                return
            stream = self._stream
            self._stream = None
            self._recording = False
            self._frames = []

        if stream is not None:
            stream.stop()
            stream.close()

    def _callback(
        self, indata: AudioFrame, frames: int, time_data: object, status: object
    ) -> None:
        del frames
        del time_data
        del status
        with self._lock:
            if self._recording:
                self._frames.append(indata.copy())

    def _ensure_audio_modules(self) -> None:
        if self._sd is not None and self._sf is not None:
            return
        try:
            sounddevice_module = importlib.import_module("sounddevice")
            soundfile_module = importlib.import_module("soundfile")
        except Exception as error:
            raise RuntimeError(
                "Audio dependencies missing. Install sounddevice and soundfile."
            ) from error

        self._sd = cast(_SoundDeviceModule, cast(object, sounddevice_module))
        self._sf = cast(_SoundFileModule, cast(object, soundfile_module))
