"""Import the pinned official SpecEdge source without editing its files."""
import importlib.util
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / 'official'
INTEGRATION = ROOT / 'integration'
UPSTREAM_COMMIT = '1edcaf02ffc41a7b57726450c5357ed216a3b9bc'


def _install_legacy_queue_logging():
    """Bridge Python 3.11 dictConfig to the official asynchronous log setup."""
    import logging
    import logging.config
    import logging.handlers
    import multiprocessing
    import log

    if getattr(log.configure_logging, '_fastsd_queue_compat', False):
        return

    def configure_logging(config):
        # Python 3.12 added QueueHandler's handlers/listener/queue dictionary
        # configuration. First let 3.11 configure the actual output handlers,
        # then connect the same handlers through the official QueueListener.
        config = copy.deepcopy(config)
        handlers = config.get('handlers', {})
        queue_specs = {name: handlers.pop(name) for name in list(handlers)
                       if handlers[name].get('class') == 'logging.handlers.QueueHandler'}
        targets = [(logging.getLogger(name), definition)
                   for name, definition in config.get('loggers', {}).items()]
        if 'root' in config:
            targets.append((logging.getLogger(), config['root']))
        original_handlers = []
        for logger, definition in targets:
            names = list(definition.get('handlers', []))
            original_handlers.append((logger, names))
            definition['handlers'] = [output for name in names
                                     for output in queue_specs.get(name, {}).get('handlers', [name])]
        logging.config.dictConfig(config)
        configured = {handler.get_name(): handler for logger, _ in original_handlers
                      for handler in logger.handlers}
        queues = {}
        for name, definition in queue_specs.items():
            if definition.get('listener') != 'log.QueueListener':
                raise ValueError('Unsupported pinned SpecEdge QueueListener configuration')
            queue_config = definition.get('queue', {})
            if queue_config.get('()') != 'multiprocessing.Queue':
                raise ValueError('Unsupported pinned SpecEdge logging queue configuration')
            queue = multiprocessing.Queue(maxsize=queue_config.get('maxsize', -1))
            handler = logging.handlers.QueueHandler(queue)
            handler.set_name(name)
            handler.listener = log.QueueListener(
                queue, *(configured[output] for output in definition['handlers']),
                respect_handler_level=definition.get('respect_handler_level', False))
            queues[name] = handler
        for logger, names in original_handlers:
            logger.handlers = [queues[name] if name in queues else configured[name] for name in names]

    configure_logging._fastsd_queue_compat = True
    log.configure_logging = configure_logging


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
    if sys.version_info < (3, 12):
        _install_legacy_queue_logging()
        compatibility.append('Python3.11 asynchronous QueueHandler dictConfig compatibility')
    return {'upstream_commit': UPSTREAM_COMMIT, 'compatibility': compatibility}
