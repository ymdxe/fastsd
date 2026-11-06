"""Import the frozen SpecEdge source with recorded Python 3.10 compatibility."""
from __future__ import annotations

import sys
import types
import typing
import atexit
import logging
import logging.handlers
import queue
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "src"
_LISTENERS = []


def _configure_legacy_logging(configuration):
    """Construct the official async log pipeline on pre-3.12 Python."""
    import log
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    file_handler = log.FileHandler(configuration["handlers"]["file"]["file_path"])
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log.ISOFormatter())
    result_handler = log.ResultHandler(configuration["handlers"]["result"]["file_path"])
    for logger, handlers in [
        (logging.getLogger(), [console, file_handler]),
        (logging.getLogger("custom_log_result"), [result_handler]),
    ]:
        messages = queue.Queue()
        handler = logging.handlers.QueueHandler(messages)
        logger.handlers[:] = [handler]
        logger.setLevel(logging.DEBUG if logger is logging.getLogger() else logging.INFO)
        if logger is not logging.getLogger():
            logger.propagate = False
        listener = logging.handlers.QueueListener(messages, *handlers, respect_handler_level=True)
        listener.start()
        _LISTENERS.append((listener, handlers))


def _close_logging():
    while _LISTENERS:
        listener, handlers = _LISTENERS.pop()
        listener.stop()
        for handler in handlers:
            handler.flush()
            handler.close()


atexit.register(_close_logging)


def install() -> None:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    if not hasattr(typing, "Self"):
        from typing_extensions import Self
        typing.Self = Self
    existing = sys.modules.get("util")
    if existing is not None:
        if Path(existing.__file__).resolve() != (VENDOR / "util.py").resolve():
            raise RuntimeError("An unrelated util module was imported before SpecEdge bootstrap")
        return
    path = VENDOR / "util.py"
    source = path.read_text()
    # Python 3.12 accepts backslashes in f-string expressions; 3.10 does not.
    # Rewrite this logging-only expression in memory, keeping vendor bytes
    # unchanged and preserving the exact string printed by the original.
    lines = source.splitlines(keepends=True)
    changed = 0
    for i, line in enumerate(lines):
        if "f\"{result.stdout.replace(" in line:
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = indent + '+ result.stdout.replace(chr(10), chr(10) + "  ")\n'
            changed += 1
    if changed != 1:
        raise RuntimeError(f"Expected one pinned util compatibility expression, found {changed}")
    module = types.ModuleType("util")
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules["util"] = module
    try:
        exec(compile("".join(lines), str(path), "exec"), module.__dict__)
        if sys.version_info < (3, 12):
            import log
            log.configure_logging = _configure_legacy_logging
    except BaseException:
        sys.modules.pop("util", None)
        raise
