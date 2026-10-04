"""Atomic replacement of local result files."""
from pathlib import Path
import os
import tempfile


def atomic_write_csv(frame, destination):
    """Keep the previous CSV intact unless its replacement is fully written.

    Atomicity is per file on filesystems supporting atomic os.replace; this
    is not a transaction across result/landmark files or a cloud durability
    guarantee. Concurrent writers to one output directory are unsupported.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='',
            prefix=f'.{destination.name}.', suffix='.tmp',
            dir=destination.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
