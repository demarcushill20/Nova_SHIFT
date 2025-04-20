"""
Provides functionality to run functions in a sandboxed subprocess.
Addresses Task P5.T2: Implement Security Sandboxing for Tool Execution.
Uses multiprocessing for isolation.
"""

import logging
import multiprocessing
import time
import threading # Added for memory monitoring thread
import psutil # Added for memory monitoring
import pickle # Added for exception check
from queue import Empty as QueueEmpty # To avoid confusion with multiprocessing.Empty
from typing import Callable, Any, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Default timeout for sandboxed execution in seconds
DEFAULT_SANDBOX_TIMEOUT = 10.0

class SandboxExecutionError(Exception):
    """Custom exception for errors during sandboxed execution."""
    pass

class SandboxTimeoutError(SandboxExecutionError):
    """Custom exception for when sandboxed execution times out."""
    pass

class SandboxMemoryError(SandboxExecutionError):
    """Custom exception for when sandboxed execution exceeds memory limits."""
    pass
# Removed duplicate class definition

def _target_wrapper(queue: multiprocessing.Queue, func: Callable, args: Tuple, kwargs: Dict[str, Any]):
    """
    Internal wrapper function executed in the subprocess.
    Calls the target function and puts the result or exception in the queue.
    """
    try:
        # TODO: Add resource limits here if possible (e.g., using `resource` module on Unix)
        # resource.setrlimit(resource.RLIMIT_CPU, (CPU_TIME_LIMIT, CPU_TIME_LIMIT))
        # resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
        # Note: Resource limits are OS-dependent and might be complex to set reliably cross-platform.
        result = func(*args, **kwargs)
        queue.put(("success", result))
    except Exception as e:
        # Capture exception details and put them in the queue
        # Avoid sending the raw exception object as it might not be pickleable
        # or could leak sensitive info. Send type and message.
        logger.warning(f"Exception caught in sandbox process: {type(e).__name__}: {e}", exc_info=False) # Log less verbosely from child
        queue.put(("error", (type(e).__name__, str(e))))

def run_in_sandbox(func: Callable,
                   args: Tuple = (),
                   kwargs: Optional[Dict[str, Any]] = None,
                   timeout: float = DEFAULT_SANDBOX_TIMEOUT,
                   memory_limit_mb: Optional[float] = None) -> Any:
    """
    Runs a function in a separate process with timeout and memory monitoring.

    Note: Memory limit is enforced by monitoring and terminating the process
          if the limit is exceeded. This is reactive and may not be perfectly precise.
          Network access is not restricted by this sandbox.

    Args:
        func: The function to execute.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        timeout: Maximum wall-clock time (in seconds) to allow the function to run.
        memory_limit_mb: Optional maximum memory usage (RSS) in megabytes.
                         If exceeded, the process is terminated.

    Returns:
        The result of the function execution.

    Raises:
        SandboxTimeoutError: If the execution exceeds the timeout.
        SandboxMemoryError: If the execution exceeds the memory limit.
        SandboxExecutionError: If the function raises an exception during execution.
        TypeError: If the function or its arguments/return value are not pickleable.
        ImportError: If psutil is required (memory_limit_mb set) but not installed.
    """
    # Check psutil dependency if memory limit is set
    if memory_limit_mb is not None:
        try:
            import psutil
        except ImportError:
            raise ImportError("psutil library is required for memory limiting but is not installed.")

    # Ensure kwargs is a dict
    if kwargs is None:
        kwargs = {}

    # Use a multiprocessing Queue for communication
    queue = multiprocessing.Queue()

    # Create the target process
    process = multiprocessing.Process(
        target=_target_wrapper,
        args=(queue, func, args, kwargs)
    )

    process.start()
    pid = process.pid
    logger.debug(f"Started sandboxed process (PID: {pid}) for function '{func.__name__}' with timeout {timeout}s"
                 f"{f' and memory limit {memory_limit_mb}MB' if memory_limit_mb else ''}.")

    # --- Memory Monitoring Thread ---
    monitor_stop_event = threading.Event()
    memory_exceeded = threading.Event() # Use Event for thread-safe flag
    monitor_exception = None # To store exception from monitor thread

    def memory_monitor_thread():
        nonlocal monitor_exception
        if memory_limit_mb is None: return # No limit set

        memory_limit_bytes = memory_limit_mb * 1024 * 1024
        try:
            p = psutil.Process(pid)
            while not monitor_stop_event.is_set():
                try:
                    mem_info = p.memory_info()
                    rss_bytes = mem_info.rss
                    if rss_bytes > memory_limit_bytes:
                        if not memory_exceeded.is_set(): # Prevent multiple logs/terminations
                            logger.warning(f"Sandboxed process {pid} for '{func.__name__}' exceeded memory limit "
                                         f"({rss_bytes / (1024*1024):.2f}MB > {memory_limit_mb:.2f}MB). Terminating.")
                            memory_exceeded.set() # Signal memory exceeded
                            try: p.terminate() # Send SIGTERM
                            except psutil.NoSuchProcess: pass
                            except Exception as term_err: logger.error(f"Error terminating process {pid} due to memory limit: {term_err}")
                        break # Stop monitoring this process
                except psutil.NoSuchProcess:
                    logger.debug(f"Process {pid} finished while monitoring memory.")
                    break # Process finished
                except Exception as poll_err:
                     logger.error(f"Error polling memory for process {pid}: {poll_err}")
                     time.sleep(0.5) # Wait before retrying poll
                     continue
                # Use wait with timeout for responsiveness to stop event
                monitor_stop_event.wait(0.2) # Check memory usage periodically
        except psutil.NoSuchProcess:
             logger.debug(f"Process {pid} finished before memory monitor could attach.")
        except Exception as e:
            logger.error(f"Memory monitor thread for process {pid} failed: {e}", exc_info=True)
            monitor_exception = e # Store exception

    monitor_thread = None
    if memory_limit_mb is not None:
        monitor_thread = threading.Thread(target=memory_monitor_thread, daemon=True)
        monitor_thread.start()
    # --- End Memory Monitoring Thread ---

    try:
        # Wait for the process to finish or timeout using join
        process.join(timeout)

        # Signal monitor thread to stop and wait for it
        monitor_stop_event.set()
        if monitor_thread:
            monitor_thread.join(timeout=1.0) # Wait briefly for monitor thread cleanup

        # Check for errors from monitor thread first
        if monitor_exception:
             raise SandboxExecutionError(f"Memory monitor failed for '{func.__name__}': {monitor_exception}") from monitor_exception

        # Check if memory limit was exceeded (flag set by monitor thread)
        if memory_exceeded.is_set():
            # Ensure process is terminated if monitor failed to do so
            if process.is_alive():
                process.terminate()
                process.join(0.1)
            raise SandboxMemoryError(f"Execution of '{func.__name__}' exceeded memory limit of {memory_limit_mb}MB.")

        # Check if process timed out (still alive after join(timeout))
        if process.is_alive():
            logger.warning(f"Sandboxed process {pid} for '{func.__name__}' timed out after {timeout}s. Terminating.")
            process.terminate()
            process.join(0.5) # Wait briefly for termination
            raise SandboxTimeoutError(f"Execution of '{func.__name__}' timed out after {timeout} seconds.")

        # Process finished normally, check queue
        try:
            result_type, result_payload = queue.get_nowait()
            if result_type == "success":
                logger.debug(f"Sandboxed process {pid} for '{func.__name__}' completed successfully.")
                return result_payload
            elif result_type == "error":
                error_type, error_msg = result_payload
                logger.error(f"Sandboxed process {pid} for '{func.__name__}' raised an exception: {error_type}: {error_msg}")
                raise SandboxExecutionError(f"Error during sandboxed execution of '{func.__name__}': {error_type}: {error_msg}")
            else:
                raise SandboxExecutionError(f"Unknown result type '{result_type}' received from sandbox for '{func.__name__}'.")

        except QueueEmpty:
            exitcode = process.exitcode
            # If finished normally (exitcode 0?) but queue is empty, it's an error.
            if exitcode == 0:
                 logger.error(f"Sandboxed process {pid} for '{func.__name__}' finished cleanly (exitcode 0) but queue is empty.")
                 raise SandboxExecutionError(f"Sandboxed execution of '{func.__name__}' finished cleanly but produced no result.")
            else:
                 # If exitcode is non-zero, it likely crashed or was terminated (e.g., by OOM killer before our monitor caught it)
                 logger.warning(f"Sandboxed process {pid} for '{func.__name__}' finished unexpectedly (exitcode: {exitcode}) with empty queue.")
                 # Raise a generic error if timeout/memory error wasn't already raised
                 if not memory_exceeded.is_set(): # Timeout error already raised if process.join timed out
                     raise SandboxExecutionError(f"Sandboxed execution of '{func.__name__}' failed unexpectedly (exitcode: {exitcode}).")
                 return None # Should be unreachable if errors are raised correctly

        except (TypeError, pickle.PicklingError) as pickle_err:
            logger.error(f"Failed to pickle arguments or return value for '{func.__name__}': {pickle_err}", exc_info=True)
            raise TypeError(f"Pickling error for '{func.__name__}': {pickle_err}") from pickle_err


    finally:
        # Final cleanup: Ensure process is terminated and queue is closed
        if process.is_alive():
            logger.warning(f"Force terminating potentially zombie sandbox process {pid} for '{func.__name__}' in finally block.")
            process.terminate()
            process.join(0.5)
        queue.close()
        queue.join_thread()
        # Ensure monitor thread is stopped if it's still running
        if monitor_thread and monitor_thread.is_alive():
            monitor_stop_event.set()
            monitor_thread.join(timeout=0.5)

# --- Example Usage ---
def _example_func(x, y=2):
    """Simple function for testing."""
    print(f"  (Inside Sandbox) Running _example_func({x}, y={y})")
    time.sleep(0.5)
    return x * y

def _example_error_func(x):
    """Function that raises an error."""
    print(f"  (Inside Sandbox) Running _example_error_func({x})")
    if x < 0:
        raise ValueError("Input cannot be negative")
    return x + 1

def _example_timeout_func(duration):
    """Function that sleeps for a specified duration."""
    print(f"  (Inside Sandbox) Running _example_timeout_func({duration}) - sleeping...")
    time.sleep(duration)
    print("  (Inside Sandbox) Sleep finished.")
    return "Slept well"

def _example_memory_func(size_mb):
    """Function that allocates memory."""
    print(f"  (Inside Sandbox) Running _example_memory_func({size_mb}MB)")
    try:
        # Allocate roughly size_mb MB of memory
        data = bytearray(int(size_mb * 1024 * 1024))
        print(f"  (Inside Sandbox) Allocated {len(data)/(1024*1024):.2f} MB")
        time.sleep(1) # Hold memory for a bit
        return f"Allocated {size_mb}MB"
    except MemoryError:
        print("  (Inside Sandbox) MemoryError caught!")
        raise SandboxExecutionError("Failed to allocate requested memory inside sandbox.")


if __name__ == "__main__":
    import time
    import pickle # Needed for exception check in run_in_sandbox

    logging.basicConfig(level=logging.DEBUG) # Enable debug logging for example

    print("\n--- Testing Sandbox Success ---")
    try:
        result = run_in_sandbox(_example_func, args=(5,), kwargs={'y': 3}, timeout=5)
        print(f"Result: {result} (Expected: 15)")
        assert result == 15
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing Sandbox Error ---")
    try:
        result = run_in_sandbox(_example_error_func, args=(-1,), timeout=5)
        print(f"Result: {result} (Should have raised error)")
    except SandboxExecutionError as e:
        print(f"Caught expected error: {e}")
        assert "ValueError: Input cannot be negative" in str(e)
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    print("\n--- Testing Sandbox Timeout ---")
    try:
        result = run_in_sandbox(_example_timeout_func, args=(3,), timeout=1.5)
        print(f"Result: {result} (Should have timed out)")
    except SandboxTimeoutError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    print("\n--- Testing Sandbox OK Timeout ---")
    try:
        result = run_in_sandbox(_example_timeout_func, args=(1,), timeout=2)
        print(f"Result: {result} (Expected: 'Slept well')")
        assert result == "Slept well"
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing Sandbox Memory Limit OK ---")
    try:
        # Allocate 50MB, limit 100MB
        result = run_in_sandbox(_example_memory_func, args=(50,), timeout=5, memory_limit_mb=100)
        print(f"Result: {result} (Expected: 'Allocated 50MB')")
        assert result == "Allocated 50MB"
    except ImportError as e:
         print(f"Skipping memory test: {e}") # Skip if psutil not installed
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing Sandbox Memory Limit Exceeded ---")
    try:
        # Allocate 150MB, limit 100MB
        result = run_in_sandbox(_example_memory_func, args=(150,), timeout=5, memory_limit_mb=100)
        print(f"Result: {result} (Should have raised memory error)")
    except ImportError as e:
         print(f"Skipping memory test: {e}") # Skip if psutil not installed
    except SandboxMemoryError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")