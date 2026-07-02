import json
import logging
import os
import socket
from urllib.request import Request, urlopen

INGESTOR_URL = os.getenv("LOGFLOW_URL", "http://localhost:3000")
API_KEY = os.getenv("LOGFLOW_API_KEY")

LEVEL_MAP = {
    50: "FATAL",
    40: "ERROR",
    30: "WARN",
    20: "INFO",
    10: "DEBUG",
    0: "DEBUG",
}


class LogflowHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.service_name = os.getenv("SERVICE_NAME", "gelo-fogo-rag")
        self.app_version = os.getenv("APP_VERSION", "1.0.0")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.host = socket.gethostname()

    def emit(self, record: logging.LogRecord) -> None:
        if not API_KEY:
            return

        try:
            body = json.dumps({
                "severity": LEVEL_MAP.get(record.levelno, "INFO"),
                "service": {
                    "name": self.service_name,
                    "version": self.app_version,
                    "environment": self.environment,
                    "host": self.host,
                },
                "message": self.format(record),
                "metadata": getattr(record, "extra", None) or {},
            }).encode()

            req = Request(
                f"{INGESTOR_URL}/api/v1/logs",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
                method="POST",
            )
            urlopen(req, timeout=2)
        except Exception:
            pass  # never block the app
