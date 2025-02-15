#
import logging
import os

PACKAGE_LOGGER_NAME = "dl_cm_logger"
_logger = logging.getLogger(PACKAGE_LOGGER_NAME)

def configure_logging(log_level: int):
    """Configure logging messages for this program."""

    # disable default logging handlers
    _logger.handlers = []
    _logger.propagate = False
    _logger.setLevel(log_level)

    console_log_handler = logging.StreamHandler()
    if log_level <= logging.DEBUG:
        console_formatter = logging.Formatter(
            fmt="[%(asctime)s] - %(levelname)s (from %(name)s in %(filename)s:%(lineno)d): %(message)s"
        )
    else:
        console_formatter = logging.Formatter(fmt="[%(asctime)s] - %(levelname)s: %(message)s")

    console_log_handler.setFormatter(console_formatter)
    console_log_handler.setLevel(log_level)
    _logger.addHandler(console_log_handler)
    return

configure_logging(logging.INFO)

def get_schema_path():
    #### Schema file path setting
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Append 'schema.yaml' to this directory
    schema_path = os.path.join(current_dir, "schema.yaml")
    return schema_path

DEFAULT_SCHEMA_PATH = get_schema_path()
