"""Import the pinned official SpecEdge source without editing its files."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / 'official'
INTEGRATION = ROOT / 'integration'
UPSTREAM_COMMIT = '1edcaf02ffc41a7b57726450c5357ed216a3b9bc'


_LEGACY_LOGGING_RESOURCES = []
_LEGACY_LOGGING_ATEXIT_REGISTERED = False


def stop_legacy_logging():
    """Drain only listeners/queues created by this compatibility wrapper."""
    while _LEGACY_LOGGING_RESOURCES:
        handler, listener, work_queue = _LEGACY_LOGGING_RESOURCES.pop()
        try:
            listener.stop()
        finally:
            work_queue.close()
            work_queue.join_thread()


def _configure_legacy_queue_logging(log_module, log_config):
    """Backport Python 3.12 QueueHandler dictConfig wiring to Python 3.11.

    Keep official output handlers, multiprocessing queues, QueueHandler and
    the official auto-starting QueueListener. Only dictConfig's construction
    and target-handler binding are performed explicitly on old Python.
    """
    import atexit
    import copy
    import logging
    import logging.config
    import multiprocessing
    global _LEGACY_LOGGING_ATEXIT_REGISTERED
    stop_legacy_logging()
    config = copy.deepcopy(log_config)
    queues = []
    for name, definition in config.get('handlers', {}).items():
        if definition.get('class') != 'logging.handlers.QueueHandler' or 'handlers' not in definition:
            continue
        target_names = list(definition.pop('handlers'))
        listener_name = definition.pop('listener', None)
        respect_levels = bool(definition.pop('respect_handler_level', False))
        queue_definition = definition.get('queue', {})
        if not isinstance(queue_definition, dict) or queue_definition.get('()') != 'multiprocessing.Queue':
            raise ValueError('Unexpected SpecEdge queue configuration; compatibility is narrowly scoped')
        work_queue = multiprocessing.Queue(maxsize=int(queue_definition.get('maxsize', -1)))
        definition['queue'] = work_queue
        queues.append((name, target_names, listener_name, respect_levels, work_queue))
    try:
        configurator = logging.config.DictConfigurator(config)
        configurator.configure()
        constructed_handlers = configurator.config['handlers']
        for name, target_names, listener_name, respect_levels, work_queue in queues:
            if listener_name not in (None, 'log.QueueListener'):
                raise ValueError('Unexpected SpecEdge QueueListener class')
            handler = constructed_handlers[name]
            targets = [constructed_handlers[target] for target in target_names]
            listener = log_module.QueueListener(work_queue, *targets,
                                                respect_handler_level=respect_levels)
            handler.listener = listener
            _LEGACY_LOGGING_RESOURCES.append((handler, listener, work_queue))
    except BaseException:
        stop_legacy_logging()
        for _, _, _, _, work_queue in queues:
            try:
                work_queue.close()
                work_queue.join_thread()
            except (ValueError, OSError):
                pass
        raise
    if not _LEGACY_LOGGING_ATEXIT_REGISTERED:
        atexit.register(stop_legacy_logging)
        _LEGACY_LOGGING_ATEXIT_REGISTERED = True


def _install_logging_compatibility():
    if sys.version_info >= (3, 12):
        return False
    import log
    if not hasattr(log, '_suite_original_configure_logging'):
        log._suite_original_configure_logging = log.configure_logging
        def configure_logging(config):
            return _configure_legacy_queue_logging(log, config)
        log.configure_logging = configure_logging
    return True

def prepare_imports():
    # multiprocessing.spawn inherits sys.path, but not imported packages.
    # Keep the suite package first so a fresh child resolves specedge.bootstrap
    # here; specedge.__path__ then exposes the untouched official subpackages.
    preferred = [str(ROOT.parent), str(INTEGRATION), str(OFFICIAL / 'src')]
    sys.path[:] = preferred + [entry for entry in sys.path if entry not in preferred]
    path = OFFICIAL / 'src' / 'util.py'
    source = path.read_text(encoding='utf-8')
    compatibility = []
    try:
        compiled = compile(source, str(path), 'exec')
    except SyntaxError:
        # Python 3.11 rejects a backslash inside an f-string expression.
        # Only this logging expression is rewritten in memory; no algorithm
        # or on-disk official source is changed.
        old = 'f"{result.stdout.replace(\'\\n\', \'\\n  \')}"'
        new = 'f"{result.stdout.replace(chr(10), chr(10) + \'  \')}"'
        if source.count(old) != 1:
            raise RuntimeError('Unsupported official util syntax; no generic code transformation applied')
        source = source.replace(old, new)
        compiled = compile(source, str(path), 'exec')
        compatibility.append('Python3.11 in-memory logging f-string compatibility')
    spec = importlib.util.spec_from_file_location('util', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['util'] = module
    exec(compiled, module.__dict__)
    if _install_logging_compatibility():
        compatibility.append('Python3.11 QueueHandler/QueueListener dictConfig wiring')
    return {'upstream_commit': UPSTREAM_COMMIT, 'compatibility': compatibility}