#!/usr/bin/env python
"""Display aggregate progress for sharded GenEval image generation."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from tqdm import tqdm


def _process_is_alive(pid: int) -> bool:
    """Check whether one local generation process still exists.

    Args:
        pid: Positive operating-system process identifier.

    Returns:
        True when the process exists or cannot be inspected due to permissions;
        false when the operating system reports that it no longer exists.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def monitor(
    image_dir: Path,
    *,
    total: int,
    pids: list[int],
    poll_interval: float,
) -> int:
    """Render aggregate progress until generation completes or all ranks exit.

    Args:
        image_dir: GenEval suite containing ``*/samples/*.png`` outputs.
        total: Expected total number of generated PNG files.
        pids: Local generation process identifiers.
        poll_interval: Seconds between filesystem progress checks.

    Returns:
        Zero when all expected images appear, or one when every process exits
        before generation is complete.
    """
    previous = 0
    with tqdm(total=total, desc="Generating GenEval images", unit="image") as progress:
        while True:
            completed = min(
                sum(1 for _ in image_dir.glob("*/samples/*.png")),
                total,
            )
            progress.update(completed - previous)
            previous = completed
            if completed >= total:
                return 0
            if not any(_process_is_alive(pid) for pid in pids):
                return 1
            time.sleep(poll_interval)


def build_parser() -> argparse.ArgumentParser:
    """Build the aggregate generation-progress command-line parser.

    Returns:
        Parser accepting an image directory, expected count, rank process IDs,
        and optional polling interval.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--pids", type=int, nargs="+", required=True)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    return parser


def main() -> int:
    """Parse arguments and monitor aggregate image generation.

    Returns:
        Process exit code from :func:`monitor`.
    """
    args = build_parser().parse_args()
    if args.total < 1 or args.poll_interval <= 0:
        raise ValueError("total and poll_interval must be positive")
    return monitor(
        args.image_dir,
        total=args.total,
        pids=args.pids,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    raise SystemExit(main())
