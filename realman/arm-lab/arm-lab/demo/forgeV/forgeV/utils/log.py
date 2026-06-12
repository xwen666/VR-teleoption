import threading
import logging
import os

from functools import wraps

try:
    import spdlog
except ImportError:
    spdlog = None


def with_logger_cache(func):
    cache = {}
    lock = threading.Lock()

    @wraps(func)
    def wrapper(name='main', size_M=10, rotate_n=5):
        with lock:
            if name in cache:
                return cache[name]

            logger = func(name, size_M, rotate_n)
            cache[name] = logger
            return logger

    return wrapper


@with_logger_cache
def get_logger_spdlog(
    name: str = 'main',
    size_m: int = 10,
    rotate_n: int = 5,
    log_level=None,
):
    if spdlog is None:
        os.makedirs('log', exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if not logger.handlers:
            formatter = logging.Formatter(
                fmt='{asctime}.{msecs:03.0f} {levelname[0]}: {message}',
                style='{',
                datefmt='%Y-%m-%d %H:%M:%S',
            )

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)

            file_handler = logging.FileHandler(f'log/{name}.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    if log_level is None:
        log_level = spdlog.LogLevel.DEBUG

    os.makedirs('log', exist_ok=True)
    console_sink = spdlog.stdout_color_sink_mt()

    file_sink = spdlog.rotating_file_sink_mt(
        filename=f'log/{name}.log',
        max_size=1024 * 1024 * size_m,
        max_files=rotate_n,
    )

    shared_logger = spdlog.SinkLogger(name=name, sinks=[console_sink, file_sink])

    shared_logger.set_level(log_level)
    shared_logger.set_pattern('%^%Y-%m-%d %T.%e %L: %v%$')

    return shared_logger


def setup_logging(name: str = 'main', loglevel: int | str = logging.DEBUG):
    formatter = logging.Formatter(
        fmt='{asctime}.{msecs:03.0f} {levelname[0]}: {message}',
        style='{',
        datefmt='%H%M%S',
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(loglevel)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(f'{name}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(loglevel)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
