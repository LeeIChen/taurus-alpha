"""Run model-generated Python for financial calculations.

WARNING: this is process isolation only (separate interpreter, temp dir,
timeout). It is NOT a security sandbox. Run it inside a container or swap in a
proper sandbox before exposing the service to untrusted input.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.return_code == 0 and not self.timed_out


def run_python(code: str, timeout_s: float = 10.0, max_output_chars: int = 20_000) -> ExecutionResult:
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],  # -I: ignore env vars and user site-packages
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={},
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                stdout=(exc.stdout or "")[:max_output_chars] if isinstance(exc.stdout, str) else "",
                stderr=f"Timed out after {timeout_s}s",
                return_code=-1,
                timed_out=True,
            )
    return ExecutionResult(
        stdout=proc.stdout[:max_output_chars],
        stderr=proc.stderr[:max_output_chars],
        return_code=proc.returncode,
    )
