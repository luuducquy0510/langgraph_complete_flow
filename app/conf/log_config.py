from typing import Any

from conf.app_config import settings


LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "verbose": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "file_handler": {
            "formatter": "default",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": settings.LOGGING_FILE_HANDLER_FILE,
            "mode": "a+",
            "maxBytes": 10*1024*1024,
            "backupCount": 0,
        },
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"level": settings.LOGGING_DEFAULT_LEVEL, "propagate": False},
        "uvicorn.error": {"level": settings.LOGGING_DEFAULT_LEVEL, "propagate": False},
    },

    "root": {
        "handlers": settings.LOGGING_LOG_HANDLER,
        "level": settings.LOGGING_DEFAULT_LEVEL,
        "propagate": False,
    },
}