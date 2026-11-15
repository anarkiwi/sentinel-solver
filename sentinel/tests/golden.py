"""Golden fixtures written atomically, so a concurrent xdist reader never sees half a file."""

import json
import os
import tempfile


def dump(path, data):
    """Write ``data`` as compact sorted JSON to ``path`` by rename; nothing is left on failure."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, separators=(",", ":"), sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
