import os
import glob
import logging

logger = logging.getLogger(__name__)

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