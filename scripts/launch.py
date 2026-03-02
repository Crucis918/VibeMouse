from __future__ import annotations

import os
import sys
import threading
import importlib
import msvcrt
from datetime import datetime
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast


def _build_tray_icon() -> object:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (64, 64), (16, 20, 28, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 56, 56), radius=14, fill=(41, 128, 185, 255))
    draw.ellipse((22, 22, 42, 42), fill=(236, 240, 241, 255))
    return image


class _TrayController:
    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._app: object | None = None
        self._worker: threading.Thread | None = None
        self._last_error: str | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._worker is not None and self._worker.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._last_error = None

        try:
            from vibemouse.config import load_config
            from vibemouse.app import VoiceMouseApp

            app = VoiceMouseApp(load_config())
        except Exception as error:
            message = f"启动 VibeMouse 引擎失败：{error}"
            self._set_last_error(message)
            _show_message("VibeMouse 启动错误", message)
            return

        with self._lock:
            worker = threading.Thread(target=self._run, args=(app,), daemon=True)
            self._app = app
            self._worker = worker
            worker.start()

    def _run(self, app: object) -> None:
        try:
            run = getattr(app, "run")
            run()
        except Exception as error:
            message = f"VibeMouse 引擎异常退出：{error}"
            self._set_last_error(message)
            _show_message("VibeMouse 运行错误", message)
        finally:
            with self._lock:
                if self._app is app:
                    self._app = None
                self._worker = None

    def stop(self) -> bool:
        with self._lock:
            app = self._app
            worker = self._worker

        if app is not None:
            request_stop = getattr(app, "request_stop", None)
            if callable(request_stop):
                request_stop()

        if worker is not None:
            worker.join(timeout=6.0)
            if worker.is_alive():
                message = "停止引擎超时：仍有后台任务未退出"
                self._set_last_error(message)
                _show_message("VibeMouse 停止错误", message)
                return False

        return True

    def restart(self) -> None:
        if not self.stop():
            return
        self.start()

    def _set_last_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error


_SINGLE_INSTANCE_FILE: Any | None = None


def _acquire_single_instance_lock(lock_path: Path) -> bool:
    global _SINGLE_INSTANCE_FILE
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fp = lock_path.open("a+")
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
        _SINGLE_INSTANCE_FILE = fp
        return True
    except OSError:
        return False


class _TeeStream:
    def __init__(self, stream: Any, log_fp: Any) -> None:
        self._stream = stream
        self._log_fp = log_fp
        self._line_buffer: str = ""

    def write(self, data: str) -> int:
        text = str(data)
        self._line_buffer += text
        lines = self._line_buffer.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            self._line_buffer = lines.pop()
        else:
            self._line_buffer = ""

        filtered: list[str] = []
        for line in lines:
            if _is_noise_log_line(line):
                continue
            filtered.append(line)

        out = "".join(filtered)
        if not out:
            return len(text)

        try:
            _ = self._stream.write(out)
        except Exception:
            pass
        _ = self._log_fp.write(out)
        return len(text)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass
        self._log_fp.flush()


def _configure_logging(log_file: Path) -> None:
    _rotate_log_file(
        log_file,
        max_bytes=_read_positive_int_env("VIBEMOUSE_LOG_MAX_BYTES", 2 * 1024 * 1024),
        backups=_read_positive_int_env("VIBEMOUSE_LOG_BACKUPS", 3),
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_file.open("a", encoding="utf-8")
    _ = log_fp.write("\n" + "=" * 70 + "\n")
    _ = log_fp.write(f"{datetime.now().isoformat()} VibeMouse launch\n")
    _ = log_fp.flush()
    sys.stdout = cast(Any, _TeeStream(sys.stdout, log_fp))
    sys.stderr = cast(Any, _TeeStream(sys.stderr, log_fp))


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return value if value > 0 else default


def _rotate_log_file(log_file: Path, *, max_bytes: int, backups: int) -> None:
    if backups <= 0:
        return
    try:
        if not log_file.exists() or log_file.stat().st_size < max_bytes:
            return
    except OSError:
        return

    for index in range(backups - 1, 0, -1):
        src = log_file.with_suffix(log_file.suffix + f".{index}")
        dst = log_file.with_suffix(log_file.suffix + f".{index + 1}")
        try:
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.replace(dst)
        except OSError:
            continue

    first = log_file.with_suffix(log_file.suffix + ".1")
    try:
        if first.exists():
            first.unlink()
        log_file.replace(first)
    except OSError:
        return


def _is_noise_log_line(line: str) -> bool:
    normalized = line.strip()
    if not normalized:
        return False
    noisy = {
        "Notice: ffmpeg is not installed. torchaudio is used to load audio",
        "If you want to use ffmpeg backend to load audio, please install it by:",
        "sudo apt install ffmpeg # ubuntu",
        "# brew install ffmpeg # mac",
    }
    return normalized in noisy


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ[key] = value.strip()


def _write_default_env_file(path: Path) -> None:
    if path.exists():
        return

    template = (
        "# VibeMouse user config\n"
        "# Edit values and restart from tray menu\n"
        "VIBEMOUSE_BACKEND=funasr_onnx\n"
        "VIBEMOUSE_DEVICE=cpu\n"
        "VIBEMOUSE_PREWARM_ON_START=true\n"
        "VIBEMOUSE_KEEP_RECORDINGS=false\n"
        "VIBEMOUSE_STOP_DELAY_MS=280\n"
        "VIBEMOUSE_LOG_MAX_BYTES=2097152\n"
        "VIBEMOUSE_LOG_BACKUPS=3\n"
        "VIBEMOUSE_FRONT_BUTTON=x2\n"
        "VIBEMOUSE_REAR_BUTTON=x1\n"
    )
    path.write_text(template, encoding="utf-8")


def _migrate_legacy_button_mapping(path: Path) -> None:
    if not path.exists():
        return

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return

    if "VIBEMOUSE_FRONT_BUTTON=x1" not in text:
        return
    if "VIBEMOUSE_REAR_BUTTON=x2" not in text:
        return

    updated = text.replace("VIBEMOUSE_FRONT_BUTTON=x1", "VIBEMOUSE_FRONT_BUTTON=x2")
    updated = updated.replace("VIBEMOUSE_REAR_BUTTON=x2", "VIBEMOUSE_REAR_BUTTON=x1")
    if updated == text:
        return

    try:
        path.write_text(updated, encoding="utf-8")
        print("Migrated side-button mapping to front=x2, rear=x1")
    except OSError:
        return


def _probe_microphone_ready() -> bool:
    try:
        sounddevice = importlib.import_module("sounddevice")
        query_devices = getattr(sounddevice, "query_devices", None)
        if not callable(query_devices):
            return False
        devices = query_devices()
    except Exception:
        return False

    if not isinstance(devices, Iterable):
        return False

    for item in devices:
        if isinstance(item, Mapping) and float(item.get("max_input_channels", 0)) > 0:
            return True
    return False


def _show_message(title: str, body: str) -> None:
    print(title)
    print(body)


def _run_first_start_guide(*, data_dir: Path, env_file: Path) -> None:
    marker = data_dir / "first-run.done"
    if marker.exists():
        return

    mic_ready = _probe_microphone_ready()
    mic_line = "Detected" if mic_ready else "Not detected"
    body = (
        "欢迎使用 VibeMouse。\n\n"
        f"麦克风检查：{mic_line}\n"
        "如果未检测到麦克风，请开启 Windows 麦克风权限。\n\n"
        "快速侧键测试：\n"
        "1) 打开记事本\n"
        "2) 按前侧键开始/停止录音\n"
        "3) 开始录音会听到上升提示音\n"
        "4) 停止录音会听到下降提示音\n"
        "5) 空闲时按后侧键发送回车\n\n"
        "如果前后侧键反了，请编辑：\n"
        f"{env_file}\n"
        "设置：\n"
        "VIBEMOUSE_FRONT_BUTTON=x1\n"
        "VIBEMOUSE_REAR_BUTTON=x2\n"
        "然后在托盘菜单里点击“重启引擎”。"
    )
    _show_message("VibeMouse 首次启动", body)
    marker.write_text("ok\n", encoding="utf-8")


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    local_app_data = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    vibemouse_data = local_app_data / "VibeMouse"
    vibemouse_data.mkdir(parents=True, exist_ok=True)
    instance_lock = vibemouse_data / "vibemouse.lock"
    if not _acquire_single_instance_lock(instance_lock):
        _show_message("VibeMouse", "VibeMouse 已在运行，无需重复启动。")
        return 0

    env_file = vibemouse_data / "vibemouse.env"
    log_file = vibemouse_data / "logs" / "vibemouse-launch.log"

    _configure_logging(log_file)
    print(f"Log file: {log_file}")

    _write_default_env_file(env_file)
    _migrate_legacy_button_mapping(env_file)
    _load_env_file(env_file)

    os.environ.setdefault("VIBEMOUSE_BACKEND", "funasr_onnx")
    os.environ.setdefault("VIBEMOUSE_DEVICE", "cpu")
    os.environ.setdefault("VIBEMOUSE_PREWARM_ON_START", "true")
    os.environ.setdefault("VIBEMOUSE_TEMP_DIR", str(vibemouse_data / "temp"))
    os.environ.setdefault(
        "VIBEMOUSE_STATUS_FILE",
        str(vibemouse_data / "vibemouse-status.json"),
    )

    if "--cli" in sys.argv:
        from vibemouse.main import main as vibemouse_main

        return vibemouse_main(["run"])

    import pystray

    controller = _TrayController()
    controller.start()
    _run_first_start_guide(data_dir=vibemouse_data, env_file=env_file)

    def on_start(icon: object, item: object) -> None:
        del icon
        del item
        controller.start()

    def on_stop(icon: object, item: object) -> None:
        del icon
        del item
        controller.stop()

    def on_restart(icon: object, item: object) -> None:
        del icon
        del item
        controller.restart()

    def on_quit(icon: object, item: object) -> None:
        del item
        controller.stop()
        cast(Any, icon).stop()

    def on_open_log_folder(icon: object, item: object) -> None:
        del icon
        del item
        log_dir = log_file.parent
        if os.name == "nt":
            try:
                os.startfile(str(log_dir))
                return
            except Exception:
                pass
        _show_message("VibeMouse 日志目录", str(log_dir))

    def on_show_last_error(icon: object, item: object) -> None:
        del icon
        del item
        message = controller.last_error or "本次会话暂无运行错误记录。"
        print("VibeMouse 状态")
        print(message)
        if os.name == "nt":
            try:
                if log_file.exists():
                    os.startfile(str(log_file))
                else:
                    os.startfile(str(log_file.parent))
                return
            except Exception:
                pass
        _show_message("VibeMouse 状态", message)

    icon = pystray.Icon(
        "VibeMouse",
        icon=_build_tray_icon(),
        title="VibeMouse",
        menu=pystray.Menu(
            pystray.MenuItem(
                lambda _item: "状态：运行中" if controller.is_running else "状态：已停止",
                None,
                enabled=False,
            ),
            pystray.MenuItem("启动引擎", on_start),
            pystray.MenuItem("停止引擎", on_stop),
            pystray.MenuItem("重启引擎", on_restart),
            pystray.MenuItem("查看最近错误", on_show_last_error),
            pystray.MenuItem("打开日志目录", on_open_log_folder),
            pystray.MenuItem("退出", on_quit),
        ),
    )
    icon.run()
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
