import logging
import sys


stream_handler = logging.StreamHandler(sys.stdout)
format_ = ('[%(asctime)s] {%(filename)s:%(lineno)d} '
           '%(levelname)s - %(message)s')

try:
    # use colored logs if installed
    import coloredlogs
    formatter = coloredlogs.ColoredFormatter(fmt=format_)
    stream_handler.setFormatter(formatter)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format=format_,
    datefmt='%m-%d %H:%M:%S',
    handlers=[stream_handler]
)
logger = logging.getLogger(__name__)
