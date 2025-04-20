"""
Implementation of the File Reader Toolkit for Nova SHIFT.

Provides the capability to read text content from local files.
"""

import logging
import os
from typing import Union
# Removed sandbox imports

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a base directory for allowed file reads (relative to project root)
# For now, let's assume agents can read from anywhere within the project
# A stricter approach would be needed in production.
# Example: ALLOWED_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'readable_data'))
ALLOWED_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Project root
def is_safe_path(requested_path: str) -> bool:
    """
    Checks if the requested path is within the allowed base directory
    and does not contain potentially unsafe elements like '..'.

    Args:
        requested_path: The file path requested by the agent.

    Returns:
        True if the path is considered safe, False otherwise.
    """
    # Normalize the path to resolve '.' and '..' if possible (on the requested path itself)
    normalized_requested_path = os.path.normpath(requested_path)

    # Prevent absolute paths
    if os.path.isabs(normalized_requested_path):
        logger.warning(f"Blocked attempt to access absolute path: {normalized_requested_path}")
        return False

    # Construct the full path relative to the allowed base directory
    # IMPORTANT: This assumes the agent provides paths relative to the project root or a known location.
    # If paths are relative to CWD, this logic needs adjustment based on where the agent runs.
    # For now, let's assume paths are relative to the project root (where NOVA_SHIFT folder is).
    # We need to be careful here. Let's enforce paths start within the project.
    # A safer way: resolve the full path and check if it starts with ALLOWED_BASE_DIR.

    # Get the absolute path of the intended target
    # Assuming the script runs from somewhere within the project structure
    # or that the path is relative to the project root defined by ALLOWED_BASE_DIR
    try:
        # Be cautious with user-provided paths. Let's join with base dir.
        full_path = os.path.abspath(os.path.join(ALLOWED_BASE_DIR, normalized_requested_path))
    except Exception as e:
        logger.error(f"Error constructing full path for {normalized_requested_path}: {e}")
        return False


    # Check if the resolved path is still within the allowed base directory
    if not full_path.startswith(ALLOWED_BASE_DIR):
        logger.warning(f"Blocked attempt to access path outside allowed base directory: {full_path} (Base: {ALLOWED_BASE_DIR})")
        return False

    # Double-check for '..' components after normalization, although abspath/startswith should handle most cases.
    if ".." in full_path.split(os.sep):
         logger.warning(f"Blocked attempt to access path containing '..': {full_path}")
         return False


    return True


# --- Synchronous Tool Function ---
def read_text_file(file_path: str) -> str:
    """
    Reads the text content of a specified local file, assuming UTF-8 encoding.

    Includes safety checks to prevent accessing files outside the project directory.

    Args:
        file_path: The relative path (from project root) to the text file to read.

    Returns:
        The content of the file as a string if successful and safe.
        An error message string if the file cannot be read, is not found,
        or the path is considered unsafe.
    """
    if not isinstance(file_path, str):
        return "Error: file_path must be a string."

    logger.info(f"Attempting to read file: {file_path}")

    # --- Path Safety Checks ---
    if os.path.isabs(file_path):
        logger.warning(f"Blocked attempt to access absolute path: {file_path}")
        return f"Error: Access denied. Absolute paths are not allowed ('{file_path}')."

    try:
        full_resolved_path = os.path.abspath(os.path.join(ALLOWED_BASE_DIR, file_path))
    except Exception as e:
        logger.error(f"Error resolving path '{file_path}' relative to base '{ALLOWED_BASE_DIR}': {e}")
        return f"Error: Could not resolve path '{file_path}'."

    base_check_path = os.path.join(ALLOWED_BASE_DIR, '')
    if not full_resolved_path.startswith(base_check_path):
        logger.warning(f"Blocked attempt to access path outside allowed base directory. Resolved: '{full_resolved_path}', Base: '{ALLOWED_BASE_DIR}'")
        return f"Error: Access denied. Path '{file_path}' resolves outside allowed directories."
    # --- End Safety Checks ---

    # Use the resolved, validated path for reading
    full_check_path = full_resolved_path

    logger.info(f"Reading file at resolved path: {full_check_path}")
    try:
        # Ensure the path actually points to a file
        if not os.path.isfile(full_check_path):
             logger.warning(f"File not found at resolved path: {full_check_path}")
             # Return specific error consistent with FileNotFoundError exception below
             return f"Error: File not found at path '{file_path}'."

        with open(full_check_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Successfully read file: {file_path} (resolved: {full_check_path})")
        return content
    except FileNotFoundError:
        # This case might be redundant due to the isfile check, but kept for robustness
        logger.error(f"File not found: {file_path} (resolved: {full_check_path})")
        return f"Error: File not found at path '{file_path}'."
    except IOError as e:
        logger.error(f"IOError reading file '{file_path}' (resolved: {full_check_path}): {e}", exc_info=True)
        return f"Error: Could not read file '{file_path}'. Reason: {e}"
    except UnicodeDecodeError as e:
        logger.error(f"UnicodeDecodeError reading file '{file_path}' (resolved: {full_check_path}): {e}. Try ensuring it's UTF-8 encoded.", exc_info=True)
        return f"Error: Could not decode file '{file_path}' as UTF-8. Ensure the file is text and UTF-8 encoded."
    except Exception as e:
        logger.error(f"An unexpected error occurred reading file '{file_path}' (resolved: {full_check_path}): {e}", exc_info=True)
        return f"Error: An unexpected error occurred while reading file '{file_path}': {e}"

# Example Usage (can be removed or moved to tests later)
if __name__ == '__main__':
    # Create dummy files for testing relative to this script's location
    test_dir = os.path.dirname(__file__)
    safe_file_rel = "test_read.txt" # Relative to script dir
    safe_file_proj_rel = os.path.join("tools", "file_reader", "test_read.txt") # Relative to project root
    unsafe_file_abs = os.path.abspath(os.path.join(test_dir, "test_read.txt")) # Absolute path
    unsafe_file_traversal = os.path.join("..", "..", "README.md") # Path traversal attempt

    # Create a safe file within the project structure for testing
    safe_file_full_path = os.path.join(test_dir, safe_file_rel)
    try:
        with open(safe_file_full_path, "w", encoding="utf-8") as f:
            f.write("This is a safe test file content.")
    except IOError:
        print(f"Could not create test file at {safe_file_full_path}")


    test_paths = [
        safe_file_proj_rel, # Safe relative path within project
        "non_existent_file.txt", # File not found
        unsafe_file_abs, # Unsafe absolute path
        unsafe_file_traversal, # Unsafe path traversal
        "../calculator/calculator_toolkit.py", # Another potentially safe relative path
        "README.md", # File at project root
        # Add a path outside the project if possible for testing is_safe_path, e.g. C:/Windows/system.ini
        # But the read function itself should block it based on the check
    ]

    print(f"Project Root used for checks: {ALLOWED_BASE_DIR}")
    print("-" * 30)

    for path in test_paths:
        print(f"Reading: '{path}'")
        content_or_error = read_text_file(path)
        if content_or_error.startswith("Error:"):
            print(content_or_error)
        else:
            print(f"Content (first 50 chars): '{content_or_error[:50]}...'")
        print("-" * 20)

    # Clean up the dummy file
    try:
        if os.path.exists(safe_file_full_path):
            os.remove(safe_file_full_path)
    except IOError:
         print(f"Could not remove test file at {safe_file_full_path}")