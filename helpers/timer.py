import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:

    _message = "Duration of {}: {:0.6f} seconds"
    _method = None

    def __init__(self):
        self._start_time = None

    def start(self, method="unspecified"):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._method = method
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(self._message.format(self._method, elapsed_time))
        return elapsed_time
