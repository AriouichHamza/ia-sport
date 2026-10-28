# services/logging_service.py
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Callable, Any, Tuple

def capture_logs(fn: Callable[..., Any], *args, **kwargs) -> Tuple[Any, str]:
    """
    Exécute fn(*args, **kwargs) en capturant tout print()/warnings sur stdout/stderr.
    Retourne (result, logs_str).
    """
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()
