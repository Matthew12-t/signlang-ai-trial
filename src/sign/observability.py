"""Safe request logging without media or transcript contents."""

import logging
from typing import Any

logger = logging.getLogger("isyara.sign")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def log_prediction(**fields: Any) -> None:
    safe_fields = {key: value for key, value in fields.items() if key not in {"video", "frames"}}
    logger.info("sign_prediction %s", safe_fields)
