import os
import glob
import time
import logging

logger = logging.getLogger(__name__)


def cleanup_old_temp_files(directory, max_age_hours=24, patterns=("*.pkl", "*.json")):
    """
    Delete temp files older than max_age_hours (by modification time).

    Per-session graph/route state is stored as pickles keyed by boundary_id and
    has no expiry, so abandoned sessions accumulate on disk indefinitely. This
    reclaims that space without touching active sessions: files for an in-flight
    request were written within the last few minutes, well under the cutoff.

    Args:
        directory: Directory to scan for temp files.
        max_age_hours: Files older than this (by mtime) are removed.
        patterns: Glob patterns of files eligible for cleanup.

    Returns:
        Number of files deleted.
    """
    if not os.path.isdir(directory):
        return 0

    cutoff = time.time() - max_age_hours * 3600
    deleted = 0
    for pattern in patterns:
        for file in glob.glob(os.path.join(directory, pattern)):
            try:
                if os.path.getmtime(file) < cutoff:
                    os.remove(file)
                    deleted += 1
                    logger.debug(f"[CLEANUP] Removed stale temp file: {file}")
            except OSError as e:
                # File may have been removed concurrently or be inaccessible.
                logger.warning(f"[CLEANUP] Could not remove {file}: {e}")

    if deleted:
        logger.info(f"[CLEANUP] Removed {deleted} temp file(s) older than {max_age_hours}h")
    return deleted


def cleanup_temp_files(directory, max_files=10):
    """
    Cleanup the temporary files in the specified directory, keeping only the latest max_files.
    """
    files = glob.glob(os.path.join(directory, "*.json"))
    if len(files) > max_files:
        # Sort files by creation time
        files.sort(key=os.path.getctime)
        # Delete the oldest files
        for file in files[:-max_files]:
            os.remove(file)
            logger.debug(f"Deleted old temp file: {file}")