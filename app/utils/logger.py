import sys
import json
from loguru import logger

def serialize_log(record):
    """Formats log records into a clean, unified JSON structure for production."""
    subset = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"]
    }
    # Inject the serialized string back into the record's extra parameters
    record["extra"]["serialized"] = json.dumps(subset)

# Clear standard formatting rules and apply our serialize patcher hook
logger.remove()
logger.add(
    sys.stdout,
    format="{extra[serialized]}"
)

# Bind the dynamic structural mapping modifier directly to our system logger export
sys_logger = logger.patch(serialize_log)